"""
Code2Graph: Chuyển đổi mã nguồn C thành đồ thị cho việc phát hiện lỗi tự động.

Công cụ này đọc file C và sinh ra file .txt/.jsonl chứa biểu diễn đồ thị 
của mã nguồn, tương thích với Train_Model.py.

Yêu cầu:
    pip install pycparser

Sử dụng:
    python Code2Graph.py --input <file.c> --output <output.txt>
    python Code2Graph.py --input-dir <folder> --output-dir <folder> --label 0/1
    python Code2Graph.py --input-dir <folder> --output <train.jsonl> --format jsonl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from pycparser import c_parser, c_ast, parse_file, c_generator
except ImportError:
    print("Vui lòng cài đặt pycparser: pip install pycparser")
    sys.exit(1)

# Optional: cho visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False


# ============================================================================
# Định nghĩa vocabulary mặc định cho node types và edge types
# ============================================================================

DEFAULT_NODE_VOCAB = {
    "ArrayDecl": 1,
    "ArrayRef": 2,
    "Assignment": 3,
    "BinaryOp": 4,
    "Break": 5,
    "Case": 6,
    "Cast": 7,
    "Compound": 8,
    "CompoundLiteral": 9,
    "Constant": 10,
    "Continue": 11,
    "Decl": 12,
    "DeclList": 13,
    "Default": 14,
    "DoWhile": 15,
    "EllipsisParam": 16,
    "EmptyStatement": 17,
    "Enum": 18,
    "Enumerator": 19,
    "EnumeratorList": 20,
    "ExprList": 21,
    "FileAST": 22,
    "For": 23,
    "FuncCall": 24,
    "FuncDecl": 25,
    "FuncDef": 26,
    "Goto": 27,
    "ID": 28,
    "IdentifierType": 29,
    "If": 30,
    "InitList": 31,
    "Label": 32,
    "NamedInitializer": 33,
    "ParamList": 34,
    "PtrDecl": 35,
    "Return": 36,
    "StaticAssert": 37,
    "Struct": 38,
    "StructRef": 39,
    "Switch": 40,
    "TernaryOp": 41,
    "TypeDecl": 42,
    "Typedef": 43,
    "Typename": 44,
    "UnaryOp": 45,
    "Union": 46,
    "While": 47,
    "Pragma": 48,
    "AlignasT": 49,
    "Unknown": 50,
}

DEFAULT_EDGE_VOCAB = {
    "ast": {
        "child": 1,
        "next_sibling": 2,
    },
    "cdfg": {
        "data_dep": 1,       # Data dependency (compute_from)
        "control_dep": 2,    # Control dependency (guarded_by)
        "control_dep_neg": 3,  # Negation control dependency
        "use_def": 4,        # Use-definition chain
        "control_flow": 5,   # Control flow (jump)
    }
}


class ASTNode:
    """Đại diện cho một node trong AST."""
    
    def __init__(self, node_id: int, node_type: str, name: str = "", 
                 coord: str = "", value: Any = None):
        self.node_id = node_id
        self.node_type = node_type
        self.name = name
        self.coord = coord
        self.value = value
        self.children: List[int] = []  # IDs of children
        self.parent: Optional[int] = None


class Code2GraphConverter:
    """Chuyển đổi mã C thành biểu diễn đồ thị."""
    
    def __init__(self, node_vocab: Dict = None, edge_vocab: Dict = None):
        self.node_vocab = node_vocab or DEFAULT_NODE_VOCAB
        self.edge_vocab = edge_vocab or DEFAULT_EDGE_VOCAB
        self.reset()
        
    def reset(self):
        """Reset trạng thái converter."""
        self.nodes: List[ASTNode] = []
        self.node_map: Dict[int, int] = {}  # pycparser node id -> our node id
        self.ast_edges: List[Tuple[int, int, int]] = []  # (src, dst, edge_type)
        self.cdfg_edges: List[Tuple[int, int, int]] = []
        self.current_id = 0
        
        # Theo dõi variable definitions và uses cho data flow
        self.var_definitions: Dict[str, List[int]] = {}  # var_name -> [node_ids]
        self.var_uses: Dict[str, List[int]] = {}  # var_name -> [node_ids]
        
        # Control flow tracking
        self.current_scope_conditions: List[int] = []  # Stack of condition node IDs
        
    def _get_node_type_id(self, node_type: str) -> int:
        """Lấy ID của node type từ vocabulary."""
        return self.node_vocab.get(node_type, self.node_vocab.get("Unknown", 50))
    
    def _create_node(self, pycparser_node: c_ast.Node) -> int:
        """Tạo một node mới và trả về ID của nó."""
        node_type = pycparser_node.__class__.__name__
        
        # Lấy thông tin bổ sung tùy theo loại node
        name = ""
        value = None
        
        if hasattr(pycparser_node, 'name') and pycparser_node.name:
            name = pycparser_node.name
        if hasattr(pycparser_node, 'value') and pycparser_node.value:
            value = pycparser_node.value
        if hasattr(pycparser_node, 'op'):
            name = pycparser_node.op
            
        coord = str(pycparser_node.coord) if pycparser_node.coord else ""
        
        node = ASTNode(
            node_id=self.current_id,
            node_type=node_type,
            name=name,
            coord=coord,
            value=value
        )
        
        self.nodes.append(node)
        self.node_map[id(pycparser_node)] = self.current_id
        self.current_id += 1
        
        return node.node_id
    
    def _add_ast_edge(self, parent_id: int, child_id: int, edge_type: str = "child"):
        """Thêm một cạnh AST."""
        edge_type_id = self.edge_vocab["ast"].get(edge_type, 1)
        self.ast_edges.append((parent_id, child_id, edge_type_id))
        
    def _add_cdfg_edge(self, src_id: int, dst_id: int, edge_type: str = "data_dep"):
        """Thêm một cạnh CDFG (Control/Data Flow Graph)."""
        edge_type_id = self.edge_vocab["cdfg"].get(edge_type, 1)
        self.cdfg_edges.append((src_id, dst_id, edge_type_id))
    
    def _visit_node(self, node: c_ast.Node, parent_id: Optional[int] = None) -> int:
        """Duyệt qua AST node và xây dựng graph."""
        if node is None:
            return -1
            
        node_id = self._create_node(node)
        
        # Thêm edge từ parent
        if parent_id is not None:
            self._add_ast_edge(parent_id, node_id, "child")
            self.nodes[node_id].parent = parent_id
            self.nodes[parent_id].children.append(node_id)
        
        # Xử lý data flow cho variables
        node_type = node.__class__.__name__
        
        # Track variable definitions
        if node_type == "Decl" and hasattr(node, 'name') and node.name:
            var_name = node.name
            if var_name not in self.var_definitions:
                self.var_definitions[var_name] = []
            self.var_definitions[var_name].append(node_id)
            
        # Track variable uses (ID nodes)
        if node_type == "ID" and hasattr(node, 'name') and node.name:
            var_name = node.name
            if var_name not in self.var_uses:
                self.var_uses[var_name] = []
            self.var_uses[var_name].append(node_id)
            
        # Track assignments
        if node_type == "Assignment" and hasattr(node, 'lvalue'):
            if isinstance(node.lvalue, c_ast.ID):
                var_name = node.lvalue.name
                if var_name not in self.var_definitions:
                    self.var_definitions[var_name] = []
                # Assignment cũng là một definition
                
        # Handle control flow conditions
        if node_type in ("If", "While", "DoWhile", "For"):
            self.current_scope_conditions.append(node_id)
        
        # Duyệt children
        children = node.children() if hasattr(node, 'children') else []
        child_ids = []
        
        for child_name, child_node in children:
            if child_node is not None:
                if isinstance(child_node, list):
                    for item in child_node:
                        if item is not None:
                            cid = self._visit_node(item, node_id)
                            if cid >= 0:
                                child_ids.append(cid)
                else:
                    cid = self._visit_node(child_node, node_id)
                    if cid >= 0:
                        child_ids.append(cid)
        
        # Add next_sibling edges between children
        for i in range(len(child_ids) - 1):
            self._add_ast_edge(child_ids[i], child_ids[i + 1], "next_sibling")
            
        # Pop condition scope
        if node_type in ("If", "While", "DoWhile", "For"):
            if self.current_scope_conditions:
                self.current_scope_conditions.pop()
        
        return node_id
    
    def _build_data_flow_edges(self):
        """Xây dựng các cạnh data flow dựa trên use-def analysis."""
        for var_name, use_ids in self.var_uses.items():
            if var_name in self.var_definitions:
                def_ids = self.var_definitions[var_name]
                for use_id in use_ids:
                    # Tìm definition gần nhất trước use
                    relevant_defs = [d for d in def_ids if d < use_id]
                    if relevant_defs:
                        last_def = max(relevant_defs)
                        self._add_cdfg_edge(last_def, use_id, "data_dep")
    
    def _build_control_flow_edges(self, ast: c_ast.Node):
        """Xây dựng control flow edges đơn giản."""
        # Duyệt lại để tìm các control flow edges
        self._visit_for_control_flow(ast)
    
    def _visit_for_control_flow(self, node: c_ast.Node, condition_id: Optional[int] = None):
        """Duyệt AST để tạo control flow edges."""
        if node is None:
            return
            
        node_type = node.__class__.__name__
        node_id = self.node_map.get(id(node))
        
        if node_id is None:
            return
            
        # Add control dependency từ condition
        if condition_id is not None and node_id != condition_id:
            self._add_cdfg_edge(condition_id, node_id, "control_dep")
        
        # Xử lý các cấu trúc điều khiển
        if node_type == "If":
            cond_id = self.node_map.get(id(node.cond)) if node.cond else None
            if node.iftrue:
                self._visit_for_control_flow(node.iftrue, cond_id)
            if node.iffalse:
                self._visit_for_control_flow(node.iffalse, cond_id)
        elif node_type == "While":
            cond_id = self.node_map.get(id(node.cond)) if node.cond else None
            if node.stmt:
                self._visit_for_control_flow(node.stmt, cond_id)
        elif node_type == "For":
            cond_id = self.node_map.get(id(node.cond)) if node.cond else None
            if node.stmt:
                self._visit_for_control_flow(node.stmt, cond_id)
        elif node_type == "DoWhile":
            cond_id = self.node_map.get(id(node.cond)) if node.cond else None
            if node.stmt:
                self._visit_for_control_flow(node.stmt, cond_id)
        else:
            # Duyệt children
            children = node.children() if hasattr(node, 'children') else []
            for child_name, child_node in children:
                if child_node is not None:
                    if isinstance(child_node, list):
                        for item in child_node:
                            self._visit_for_control_flow(item, condition_id)
                    else:
                        self._visit_for_control_flow(child_node, condition_id)
    
    def filter_user_code_only(self, start_line: int = 25):
        """
        Lọc bỏ các nodes từ fake typedefs, chỉ giữ lại code của người dùng.
        
        Args:
            start_line: Dòng bắt đầu của code người dùng (sau fake typedefs)
        """
        # Xác định nodes cần giữ
        keep_nodes = set()
        for node in self.nodes:
            if node.coord:
                # Lấy line number từ coord
                try:
                    line_str = node.coord.split(':')[1] if ':' in node.coord else '0'
                    line_num = int(line_str)
                    if line_num >= start_line:
                        keep_nodes.add(node.node_id)
                except:
                    pass
            elif node.node_type == "FileAST":
                # Giữ root node nhưng sẽ được xử lý riêng
                pass
        
        if not keep_nodes:
            return  # Không có gì để lọc
        
        # Tạo mapping từ old id sang new id
        old_to_new = {}
        new_nodes = []
        
        # Tìm node gốc mới (thường là FuncDef)
        root_candidates = [n for n in self.nodes if n.node_id in keep_nodes and n.node_type == "FuncDef"]
        if not root_candidates:
            root_candidates = [n for n in self.nodes if n.node_id in keep_nodes]
        
        if root_candidates:
            # BFS từ root để lấy tất cả nodes con
            from collections import deque
            
            # Xây dựng đồ thị parent->children
            children_map = {n.node_id: [] for n in self.nodes}
            for src, dst, _ in self.ast_edges:
                if src in children_map:
                    children_map[src].append(dst)
            
            # Tìm tất cả descendants của các root
            all_keep = set()
            for root in root_candidates:
                queue = deque([root.node_id])
                while queue:
                    curr = queue.popleft()
                    if curr in all_keep:
                        continue
                    all_keep.add(curr)
                    queue.extend(children_map.get(curr, []))
            
            keep_nodes = all_keep
        
        # Lọc và reindex nodes
        sorted_keeps = sorted(keep_nodes)
        for new_id, old_id in enumerate(sorted_keeps):
            old_to_new[old_id] = new_id
            old_node = self.nodes[old_id]
            new_node = ASTNode(
                node_id=new_id,
                node_type=old_node.node_type,
                name=old_node.name,
                coord=old_node.coord,
                value=old_node.value
            )
            new_nodes.append(new_node)
        
        # Lọc và reindex edges
        new_ast_edges = []
        for src, dst, etype in self.ast_edges:
            if src in old_to_new and dst in old_to_new:
                new_ast_edges.append((old_to_new[src], old_to_new[dst], etype))
        
        new_cdfg_edges = []
        for src, dst, etype in self.cdfg_edges:
            if src in old_to_new and dst in old_to_new:
                new_cdfg_edges.append((old_to_new[src], old_to_new[dst], etype))
        
        # Update children và parent references
        for node in new_nodes:
            node.children = []
            node.parent = None
        
        for src, dst, etype in new_ast_edges:
            if etype == 1:  # child edge
                new_nodes[src].children.append(dst)
                new_nodes[dst].parent = src
        
        # Cập nhật
        self.nodes = new_nodes
        self.ast_edges = new_ast_edges
        self.cdfg_edges = new_cdfg_edges
    
    def parse_code(self, code: str, filename: str = "<unknown>", 
                   filter_typedefs: bool = True) -> bool:
        """Parse mã C từ string."""
        self.reset()
        
        # Preprocess: thêm fake includes và xử lý một số vấn đề phổ biến
        preprocessed = self._preprocess_code(code)
        
        try:
            parser = c_parser.CParser()
            ast = parser.parse(preprocessed, filename=filename)
            
            # Xây dựng AST graph
            self._visit_node(ast)
            
            # Xây dựng data flow edges  
            self._build_data_flow_edges()
            
            # Xây dựng control flow edges
            self._build_control_flow_edges(ast)
            
            # Lọc bỏ fake typedefs nếu cần
            if filter_typedefs:
                self.filter_user_code_only(start_line=25)
            
            return True
            
        except Exception as e:
            print(f"Lỗi khi parse {filename}: {e}")
            return False
    
    def parse_file(self, filepath: str) -> bool:
        """Parse mã C từ file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            return self.parse_code(code, filename=filepath)
        except Exception as e:
            print(f"Lỗi khi đọc file {filepath}: {e}")
            return False
    
    def _preprocess_code(self, code: str) -> str:
        """Tiền xử lý code để pycparser có thể parse."""
        # Loại bỏ các directives phức tạp
        lines = code.split('\n')
        processed_lines = []
        in_multiline_directive = False
        
        for line in lines:
            stripped = line.strip()
            
            # Xử lý multiline preprocessor directives
            if in_multiline_directive:
                if not stripped.endswith('\\'):
                    in_multiline_directive = False
                continue
            
            # Bỏ qua hoặc đơn giản hóa các preprocessor directives
            if stripped.startswith('#'):
                if stripped.endswith('\\'):
                    in_multiline_directive = True
                continue
            
            processed_lines.append(line)
        
        code = '\n'.join(processed_lines)
        
        # Thêm typedef cơ bản để pycparser có thể parse
        fake_typedefs = """
typedef int size_t;
typedef int ssize_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
typedef int int32_t;
typedef long int64_t;
typedef unsigned char uint8_t;
typedef char int8_t;
typedef unsigned short uint16_t;
typedef short int16_t;
typedef void* FILE;
typedef int bool;
typedef int BOOL;
typedef unsigned int DWORD;
typedef unsigned short WORD;
typedef unsigned char BYTE;
typedef char* LPSTR;
typedef const char* LPCSTR;
typedef void* LPVOID;
typedef void* HANDLE;
typedef long LONG;
typedef unsigned long ULONG;
"""
        return fake_typedefs + code
    
    def to_dict(self, label: int = 0, file_id: str = "") -> Dict:
        """Chuyển đổi graph thành dictionary tương thích với Train_Model.py."""
        node_types = [self._get_node_type_id(n.node_type) for n in self.nodes]
        
        return {
            "id": file_id,
            "label": label,
            "num_nodes": len(self.nodes),
            "node_types": node_types,
            "ast_edges": list(self.ast_edges),
            "cdfg_edges": list(self.cdfg_edges),
        }
    
    def to_json(self, label: int = 0, file_id: str = "", indent: int = None) -> str:
        """Chuyển đổi graph thành JSON string."""
        return json.dumps(self.to_dict(label, file_id), indent=indent)
    
    def to_text(self, include_details: bool = True) -> str:
        """Chuyển đổi graph thành định dạng text dễ đọc."""
        lines = []
        
        # Header
        lines.append(f"=== Graph Summary ===")
        lines.append(f"Total nodes: {len(self.nodes)}")
        lines.append(f"AST edges: {len(self.ast_edges)}")
        lines.append(f"CDFG edges: {len(self.cdfg_edges)}")
        lines.append("")
        
        # Nodes
        lines.append("=== Nodes ===")
        for node in self.nodes:
            type_id = self._get_node_type_id(node.node_type)
            if include_details:
                lines.append(f"[{node.node_id}] {node.node_type}({type_id}) name='{node.name}' coord={node.coord}")
            else:
                lines.append(f"[{node.node_id}] {node.node_type}({type_id})")
        lines.append("")
        
        # AST Edges
        lines.append("=== AST Edges ===")
        edge_type_names = {v: k for k, v in self.edge_vocab["ast"].items()}
        for src, dst, etype in self.ast_edges:
            etype_name = edge_type_names.get(etype, str(etype))
            lines.append(f"{src} --[{etype_name}]--> {dst}")
        lines.append("")
        
        # CDFG Edges
        lines.append("=== CDFG Edges ===")
        edge_type_names = {v: k for k, v in self.edge_vocab["cdfg"].items()}
        for src, dst, etype in self.cdfg_edges:
            etype_name = edge_type_names.get(etype, str(etype))
            lines.append(f"{src} --[{etype_name}]--> {dst}")
        
        return '\n'.join(lines)
    
    def save_vocab(self, output_dir: str):
        """Lưu vocabulary files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "node_vocab.json", 'w', encoding='utf-8') as f:
            json.dump(self.node_vocab, f, indent=2)
            
        with open(output_path / "edge_vocab.json", 'w', encoding='utf-8') as f:
            json.dump(self.edge_vocab, f, indent=2)
    
    def visualize(self, output_path: str = None, title: str = None,
                  figsize: Tuple[int, int] = (16, 12), show: bool = True,
                  show_legend: bool = True, node_size: int = 2500,
                  font_size: int = 9, layout: str = "hierarchical"):
        """
        Vẽ đồ thị với các loại edge và node được phân biệt bằng màu sắc.
        
        Args:
            output_path: Đường dẫn file để lưu hình (None = không lưu)
            title: Tiêu đề của đồ thị
            figsize: Kích thước figure (width, height)
            show: Hiển thị đồ thị sau khi vẽ
            show_legend: Hiển thị chú thích
            node_size: Kích thước node
            font_size: Cỡ chữ
            layout: Kiểu bố cục ("hierarchical", "spring", "kamada_kawai", "circular")
        """
        if not HAS_VISUALIZATION:
            print("Cần cài đặt matplotlib và networkx: pip install matplotlib networkx")
            return
        
        if len(self.nodes) == 0:
            print("Không có nodes để visualize")
            return
        
        # Tạo đồ thị NetworkX
        G = nx.DiGraph()
        
        # Màu sắc cho các loại node
        node_colors_map = {
            # Function related - Hồng/Đỏ
            "FuncDef": "#FF6B6B",
            "FuncDecl": "#FF8E8E",
            "FuncCall": "#FFB3B3",
            # Control structures - Cam
            "If": "#FFA500",
            "While": "#FFB347",
            "For": "#FFCC80",
            "DoWhile": "#FFD699",
            "Switch": "#FFE4B3",
            # Statements - Xám
            "Compound": "#C0C0C0",
            "Return": "#A9A9A9",
            "ExprList": "#D3D3D3",
            # Declarations - Vàng nhạt
            "Decl": "#FFFACD",
            "ParamList": "#FAFAD2",
            "TypeDecl": "#FFFFE0",
            # Expressions/Operations - Vàng đậm
            "BinaryOp": "#FFD700",
            "UnaryOp": "#FFDF00",
            "Assignment": "#FFE135",
            "TernaryOp": "#FFEA00",
            # Identifiers/Variables - Xanh nhạt
            "ID": "#ADD8E6",
            "Constant": "#B0E0E6",
            # Types - Xanh lục
            "IdentifierType": "#90EE90",
            # Default - Xám nhạt
            "default": "#E8E8E8"
        }
        
        # Thêm nodes
        for node in self.nodes:
            label = self._get_node_display_label(node)
            color = node_colors_map.get(node.node_type, node_colors_map["default"])
            G.add_node(node.node_id, label=label, node_type=node.node_type, color=color)
        
        # Phân loại edges
        ast_child_edges = []      # Đường đen liền - cấu trúc AST
        next_token_edges = []     # Đường xanh nét đứt - sequence
        data_flow_edges = []      # Đường đỏ nét đứt - data flow
        control_flow_edges = []   # Đường tím nét đứt - control flow
        
        for src, dst, etype in self.ast_edges:
            if etype == 1:  # child
                ast_child_edges.append((src, dst))
            elif etype == 2:  # next_sibling/next_token
                next_token_edges.append((src, dst))
        
        for src, dst, etype in self.cdfg_edges:
            if etype == 1:  # data_dep
                data_flow_edges.append((src, dst))
            elif etype in (2, 3, 5):  # control_dep, control_dep_neg, control_flow
                control_flow_edges.append((src, dst))
        
        # Thêm tất cả edges vào graph
        G.add_edges_from(ast_child_edges)
        G.add_edges_from(next_token_edges)
        G.add_edges_from(data_flow_edges)
        G.add_edges_from(control_flow_edges)
        
        # Tính toán layout
        pos = self._compute_layout(G, layout)
        
        # Vẽ đồ thị
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Lấy màu nodes
        node_colors = [G.nodes[n].get('color', '#E8E8E8') for n in G.nodes()]
        
        # Vẽ nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=node_size, edgecolors='black', linewidths=1.5)
        
        # Vẽ labels
        labels = {n: G.nodes[n].get('label', str(n)) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=font_size,
                                font_weight='bold')
        
        # Vẽ các loại edges khác nhau
        # 1. AST Child edges - đường đen liền
        if ast_child_edges:
            nx.draw_networkx_edges(G, pos, edgelist=ast_child_edges, ax=ax,
                                   edge_color='black', width=2.0, alpha=0.8,
                                   arrows=True, arrowsize=20, arrowstyle='-|>',
                                   connectionstyle='arc3,rad=0.1')
        
        # 2. NextToken edges - đường xanh nét đứt
        if next_token_edges:
            nx.draw_networkx_edges(G, pos, edgelist=next_token_edges, ax=ax,
                                   edge_color='#0066CC', width=1.5, alpha=0.7,
                                   style='dashed', arrows=True, arrowsize=15,
                                   arrowstyle='-|>', connectionstyle='arc3,rad=0.2')
        
        # 3. Data Flow edges - đường đỏ nét đứt
        if data_flow_edges:
            nx.draw_networkx_edges(G, pos, edgelist=data_flow_edges, ax=ax,
                                   edge_color='#CC0000', width=1.5, alpha=0.7,
                                   style='dashed', arrows=True, arrowsize=15,
                                   arrowstyle='-|>', connectionstyle='arc3,rad=0.3')
        
        # 4. Control Flow edges - đường tím nét đứt
        if control_flow_edges:
            nx.draw_networkx_edges(G, pos, edgelist=control_flow_edges, ax=ax,
                                   edge_color='#9933CC', width=1.5, alpha=0.7,
                                   style='dotted', arrows=True, arrowsize=15,
                                   arrowstyle='-|>', connectionstyle='arc3,rad=0.25')
        
        # Thêm legend
        if show_legend:
            legend_elements = [
                mpatches.Patch(facecolor='black', label='AST Child (Structure)', linewidth=2),
                mpatches.Patch(facecolor='#0066CC', label='NextToken (Sequence)', linewidth=2),
                mpatches.Patch(facecolor='#CC0000', label='Data Flow (Semantic)', linewidth=2),
                mpatches.Patch(facecolor='#9933CC', label='Control Flow', linewidth=2),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                     framealpha=0.9, title='Edge Types')
        
        # Thiết lập tiêu đề
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Lưu file nếu có output_path
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Đã lưu đồ thị vào: {output_path}")
        
        # Hiển thị
        if show:
            plt.show()
        else:
            plt.close()
    
    def _get_node_display_label(self, node: ASTNode) -> str:
        """Tạo label hiển thị cho node."""
        type_name = node.node_type
        
        # Rút gọn tên một số node types
        type_abbrev = {
            "FuncDef": "FuncDef",
            "FuncDecl": "FuncDecl",
            "FuncCall": "FuncCall",
            "Compound": "CompoundStmt",
            "Return": "ReturnStmt",
            "BinaryOp": "BinaryOp",
            "UnaryOp": "UnaryOp",
            "ParamList": "ParamList",
            "Decl": "Decl",
            "TypeDecl": "TypeDecl",
            "IdentifierType": "Type",
            "ID": "Id",
            "Constant": "Const",
        }
        
        display_type = type_abbrev.get(type_name, type_name)
        
        # Thêm thông tin bổ sung
        if node.name:
            if type_name == "BinaryOp":
                return f"{display_type}\n({node.name})"
            elif type_name == "ID":
                return f"Identifier:\n{node.name}"
            elif type_name in ("FuncDef", "FuncDecl", "FuncCall"):
                return f"{display_type}:\n{node.name}"
            elif type_name == "Decl":
                return f"Decl:\n{node.name}"
            else:
                return f"{display_type}:\n{node.name}"
        elif node.value is not None:
            return f"{display_type}:\n{node.value}"
        else:
            return display_type
    
    def _compute_layout(self, G: 'nx.DiGraph', layout_type: str) -> Dict:
        """Tính toán vị trí các node trong đồ thị."""
        if len(G.nodes()) == 0:
            return {}
        
        if layout_type == "hierarchical":
            # Sử dụng layout phân cấp theo thứ tự BFS từ root
            return self._hierarchical_layout(G)
        elif layout_type == "spring":
            return nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout_type == "kamada_kawai":
            return nx.kamada_kawai_layout(G)
        elif layout_type == "circular":
            return nx.circular_layout(G)
        else:
            return nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    def _hierarchical_layout(self, G: 'nx.DiGraph') -> Dict:
        """Tính toán layout phân cấp cho đồ thị."""
        pos = {}
        
        # Tìm root nodes (nodes không có parent trong AST child edges)
        child_edges = [(s, d) for s, d, t in self.ast_edges if t == 1]
        children = set(d for s, d in child_edges)
        parents = set(s for s, d in child_edges)
        roots = parents - children
        
        if not roots:
            roots = {0} if 0 in G.nodes() else set(list(G.nodes())[:1])
        
        # BFS để xác định level của từng node
        levels = {}
        for root in roots:
            self._assign_levels(G, root, 0, levels, child_edges)
        
        # Gán level cho các node còn lại
        for node in G.nodes():
            if node not in levels:
                levels[node] = max(levels.values()) + 1 if levels else 0
        
        # Tính vị trí x cho từng level
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
        
        # Sắp xếp nodes trong mỗi level
        max_level = max(level_nodes.keys()) if level_nodes else 0
        
        for level, nodes in level_nodes.items():
            # Sắp xếp theo node_id để có thứ tự nhất quán
            nodes.sort()
            width = len(nodes)
            for i, node in enumerate(nodes):
                # Tính x: phân bố đều, y: theo level (đảo ngược để root ở trên)
                x = (i - (width - 1) / 2) * 2.5
                y = (max_level - level) * 2
                pos[node] = (x, y)
        
        return pos
    
    def _assign_levels(self, G: 'nx.DiGraph', node: int, level: int, 
                       levels: Dict, child_edges: List):
        """Gán level cho nodes bằng BFS."""
        if node in levels:
            return
        
        levels[node] = level
        
        # Tìm children
        children = [d for s, d in child_edges if s == node]
        for child in children:
            self._assign_levels(G, child, level + 1, levels, child_edges)


def process_single_file(converter: Code2GraphConverter, input_path: str, 
                        output_path: str, label: int = 0, 
                        format: str = "text", verbose: bool = True) -> bool:
    """Xử lý một file C và lưu kết quả."""
    if verbose:
        print(f"Processing: {input_path}")
    
    if not converter.parse_file(input_path):
        return False
    
    file_id = Path(input_path).name
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if format == "jsonl" or format == "json":
            f.write(converter.to_json(label=label, file_id=file_id))
        else:
            f.write(converter.to_text())
    
    if verbose:
        print(f"  -> Saved to: {output_path}")
        print(f"     Nodes: {len(converter.nodes)}, AST edges: {len(converter.ast_edges)}, CDFG edges: {len(converter.cdfg_edges)}")
    
    return True


def process_directory(converter: Code2GraphConverter, input_dir: str, 
                      output_path: str, label: int = 0, 
                      format: str = "jsonl", verbose: bool = True) -> int:
    """
    Xử lý tất cả file .c trong thư mục.
    
    Nếu format='jsonl', tất cả output sẽ được ghi vào một file.
    Nếu format='text', mỗi file sẽ có output riêng.
    """
    input_path = Path(input_dir)
    c_files = list(input_path.glob("**/*.c"))
    
    if verbose:
        print(f"Found {len(c_files)} C files in {input_dir}")
    
    successful = 0
    
    if format == "jsonl":
        # Ghi tất cả vào một file jsonl
        with open(output_path, 'w', encoding='utf-8') as f:
            for c_file in c_files:
                if converter.parse_file(str(c_file)):
                    file_id = c_file.name
                    json_line = converter.to_json(label=label, file_id=file_id)
                    f.write(json_line + '\n')
                    successful += 1
                    if verbose:
                        print(f"  Processed: {c_file.name} ({len(converter.nodes)} nodes)")
    else:
        # Mỗi file có output riêng
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for c_file in c_files:
            out_file = output_dir / (c_file.stem + ".txt")
            if process_single_file(converter, str(c_file), str(out_file), 
                                   label=label, format="text", verbose=verbose):
                successful += 1
    
    if verbose:
        print(f"\nProcessed {successful}/{len(c_files)} files successfully")
    
    return successful


def create_dataset_splits(input_dir: str, output_dir: str, 
                          train_ratio: float = 0.7, val_ratio: float = 0.15,
                          label_from_path: bool = True, verbose: bool = True):
    """
    Tạo train/val/test splits từ thư mục chứa các file C.
    
    Nếu label_from_path=True, sẽ cố gắng lấy label từ tên thư mục (good/bad, vuln/safe, etc.)
    """
    import random
    
    converter = Code2GraphConverter()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Tìm tất cả file C
    c_files = list(input_path.glob("**/*.c"))
    random.shuffle(c_files)
    
    if verbose:
        print(f"Found {len(c_files)} C files")
    
    # Chia splits
    n_train = int(len(c_files) * train_ratio)
    n_val = int(len(c_files) * val_ratio)
    
    train_files = c_files[:n_train]
    val_files = c_files[n_train:n_train + n_val]
    test_files = c_files[n_train + n_val:]
    
    if verbose:
        print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    
    def get_label(filepath: Path) -> int:
        """Xác định label từ đường dẫn file."""
        path_lower = str(filepath).lower()
        # Các pattern cho vulnerable/bad code
        if any(x in path_lower for x in ['bad', 'vuln', 'vulnerable', 'cwe-', 'unsafe']):
            return 1
        # Các pattern cho safe/good code  
        if any(x in path_lower for x in ['good', 'safe', 'patched', 'fixed']):
            return 0
        # Mặc định
        return 0
    
    def process_split(files: List[Path], output_file: str):
        with open(output_path / output_file, 'w', encoding='utf-8') as f:
            for c_file in files:
                if converter.parse_file(str(c_file)):
                    label = get_label(c_file) if label_from_path else 0
                    json_line = converter.to_json(label=label, file_id=c_file.name)
                    f.write(json_line + '\n')
    
    process_split(train_files, "train.jsonl")
    process_split(val_files, "val.jsonl")
    process_split(test_files, "test.jsonl")
    
    # Lưu vocab
    converter.save_vocab(str(output_path))
    
    if verbose:
        print(f"Dataset saved to {output_path}")
        print(f"Files created: train.jsonl, val.jsonl, test.jsonl, node_vocab.json, edge_vocab.json")


def main():
    parser = argparse.ArgumentParser(
        description="Chuyển đổi mã nguồn C thành đồ thị cho việc phát hiện lỗi.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Xử lý một file C, output text
  python Code2Graph.py --input example.c --output example_graph.txt

  # Xử lý một file C, output JSON  
  python Code2Graph.py --input example.c --output example.json --format json --label 1

  # Xử lý thư mục và tạo file JSONL để train
  python Code2Graph.py --input-dir ./c_files --output train_data.jsonl --format jsonl --label 1

  # Tạo dataset splits (train/val/test)
  python Code2Graph.py --input-dir ./dataset --output-dir ./processed --create-splits
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                             help='Đường dẫn đến file C đầu vào')
    input_group.add_argument('--input-dir', type=str,
                             help='Đường dẫn đến thư mục chứa các file C')
    
    # Output options
    parser.add_argument('--output', '-o', type=str,
                        help='Đường dẫn file output (hoặc thư mục nếu --format text với --input-dir)')
    parser.add_argument('--output-dir', type=str,
                        help='Thư mục output cho dataset splits')
    
    # Processing options
    parser.add_argument('--format', '-f', type=str, default='text',
                        choices=['text', 'json', 'jsonl'],
                        help='Định dạng output (default: text)')
    parser.add_argument('--label', '-l', type=int, default=0,
                        help='Label cho samples (0=safe, 1=vulnerable)')
    parser.add_argument('--create-splits', action='store_true',
                        help='Tạo train/val/test splits từ thư mục input')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Tỷ lệ training set (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Tỷ lệ validation set (default: 0.15)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Không in thông tin chi tiết')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Vẽ đồ thị và lưu thành hình ảnh')
    parser.add_argument('--graph-output', type=str, default=None,
                        help='Đường dẫn file hình ảnh output cho visualization')
    parser.add_argument('--layout', type=str, default='hierarchical',
                        choices=['hierarchical', 'spring', 'kamada_kawai', 'circular'],
                        help='Kiểu layout cho đồ thị (default: hierarchical)')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    converter = Code2GraphConverter()
    
    if args.create_splits:
        if not args.input_dir:
            parser.error("--create-splits yêu cầu --input-dir")
        if not args.output_dir:
            parser.error("--create-splits yêu cầu --output-dir")
        
        create_dataset_splits(
            args.input_dir, args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            verbose=verbose
        )
    elif args.input:
        # Xử lý một file
        if not args.output:
            # Tạo output path từ input
            input_path = Path(args.input)
            if args.format == 'json' or args.format == 'jsonl':
                args.output = str(input_path.with_suffix('.json'))
            else:
                args.output = str(input_path.with_suffix('.txt'))
        
        success = process_single_file(
            converter, args.input, args.output,
            label=args.label, format=args.format, verbose=verbose
        )
        
        if not success:
            sys.exit(1)
        
        # Visualize nếu được yêu cầu
        if args.visualize:
            graph_output = args.graph_output
            if not graph_output:
                input_path = Path(args.input)
                graph_output = str(input_path.with_suffix('.png'))
            
            # Parse lại để có dữ liệu
            converter.parse_file(args.input)
            title = f"Graph Representation of '{Path(args.input).name}'"
            converter.visualize(output_path=graph_output, title=title,
                              layout=args.layout, show=True)
            
    elif args.input_dir:
        # Xử lý thư mục
        if not args.output:
            if args.format == 'jsonl':
                args.output = "output.jsonl"
            else:
                args.output = "output_graphs"
        
        successful = process_directory(
            converter, args.input_dir, args.output,
            label=args.label, format=args.format, verbose=verbose
        )
        
        if successful == 0:
            print("Không có file nào được xử lý thành công!")
            sys.exit(1)


if __name__ == "__main__":
    # Demo mode nếu chạy trực tiếp không có arguments
    if len(sys.argv) == 1:
        print("Code2Graph - Chuyển đổi mã C thành đồ thị")
        print("=" * 50)
        
        # Demo với một đoạn code mẫu
        demo_code = '''
int sum(int a, int b) {
    return a + b;
}
'''
        print("Demo code:")
        print(demo_code)
        print("\n" + "=" * 50)
        
        converter = Code2GraphConverter()
        if converter.parse_code(demo_code, "<demo>"):
            print("\nGraph representation (text format):")
            print(converter.to_text(include_details=True))
            
            print("\n" + "=" * 50)
            print("\nJSON format (cho training):")
            print(converter.to_json(label=1, file_id="demo.c", indent=2))
            
            # Hiển thị đồ thị nếu có matplotlib
            if HAS_VISUALIZATION:
                print("\n" + "=" * 50)
                print("\nĐang vẽ đồ thị...")
                title = "Graph Representation of 'int sum(int a, int b) { return a + b; }'"
                converter.visualize(output_path="demo_graph.png", title=title, show=True)
            else:
                print("\nĐể visualize đồ thị, cài đặt: pip install matplotlib networkx")
        else:
            print("Không thể parse demo code")
        
        print("\n" + "=" * 50)
        print("\nSử dụng: python Code2Graph.py --help để xem các options")
        print("\nVí dụ visualize:")
        print("  python Code2Graph.py --input example.c --visualize")
        print("  python Code2Graph.py --input example.c --visualize --layout spring")
    else:
        main()
