import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from Train_Model import VulnDetectorGraphTransformer


@dataclass
class GraphSample:
    node_types: List[int]
    node_names: List[str]
    ast_edges: List[Tuple[int, int, int]]
    cdfg_edges: List[Tuple[int, int, int]]


BUILTIN_TYPES = {
    "void",
    "char",
    "short",
    "int",
    "long",
    "float",
    "double",
    "signed",
    "unsigned",
    "_Bool",
    "bool",
}


KEYWORD_NODE_MAP = {
    "if": "IfStatement",
    "for": "ForStatement",
    "while": "WhileStatement",
    "do": "DoStatement",
    "switch": "SwitchStatement",
    "case": "CaseStatement",
    "default": "DefaultStatement",
    "break": "BreakStatement",
    "continue": "ContinueStatement",
    "return": "ReturnStatement",
    "goto": "GotoStatement",
}


TOKEN_PATTERN = re.compile(
    r"""
    (?P<identifier>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<number>0x[0-9A-Fa-f]+|\d+)
    | (?P<string>"(\\.|[^"])*"|'(\\.|[^'])*')
    | (?P<operator>==|!=|<=|>=|\+\+|--|->|&&|\|\||[+\-*/%<>=!&|^~])
    | (?P<punctuation>[(){}\[\],;])
    """,
    re.VERBOSE,
)


def strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = re.sub(r"//.*", "", source)
    source = re.sub(r"^\s*#.*$", "", source, flags=re.MULTILINE)
    return source


def safe_torch_load(model_path: Path, device: torch.device) -> dict:
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        message = str(exc)
        if "weights_only" not in message and "Unsupported" not in message:
            raise
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(
            "[WARN] weights_only=True failed; loading checkpoint with weights_only=False. "
            "Use trusted files only."
        )
        return checkpoint


def load_checkpoint(model_path: Path, device: torch.device) -> Tuple[dict, dict, dict, dict]:
    checkpoint = safe_torch_load(model_path, device)
    node_vocab = checkpoint.get("node_vocab")
    edge_vocab = checkpoint.get("edge_vocab")
    config = checkpoint.get("config", {})
    state = checkpoint.get("model_state")
    if state is None:
        raise SystemExit(f"Missing model_state in checkpoint: {model_path}")
    if not node_vocab or not edge_vocab:
        raise SystemExit(
            "Checkpoint missing node_vocab or edge_vocab. Re-export the model with vocabularies."
        )
    return state, node_vocab, edge_vocab, config


def resolve_model_path(model_dir: Path, model_path: Optional[Path]) -> Path:
    if model_path:
        return model_path
    default_path = model_dir / "Auto_Bug_Detector.pt"
    if default_path.exists():
        return default_path
    fallback = model_dir / "Auto_Bug_Detector_best.pt"
    if fallback.exists():
        return fallback
    raise SystemExit(f"Could not find model checkpoint in {model_dir}")


def vocab_size(vocab: Dict[str, int]) -> int:
    return max(vocab.values()) if vocab else 0


def edge_type_count(vocab: Dict[str, int]) -> int:
    return max(vocab.values()) if vocab else 0


def map_unknown(node_vocab: Dict[str, int]) -> int:
    return node_vocab.get("ProblemStatement", 0)


def build_cdfg_edges(
    node_names: List[str],
    ast_edges: List[Tuple[int, int, int]],
    edge_vocab: Dict[str, Dict[str, int]],
) -> List[Tuple[int, int, int]]:
    cdfg_map = edge_vocab.get("cdfg", {})
    compute_from_id = cdfg_map.get("compute_from", 1)
    expr_nodes = {
        "BinaryExpression",
        "UnaryExpression",
        "CastExpression",
        "ConditionalExpression",
        "FunctionCallExpression",
        "ArraySubscriptExpression",
    }
    cdfg_edges = []
    for src, dst, _edge_type in ast_edges:
        if node_names[src] in expr_nodes:
            cdfg_edges.append((src, dst, compute_from_id))
    return cdfg_edges


def parse_with_pycparser(
    source: str,
    node_vocab: Dict[str, int],
    edge_vocab: Dict[str, Dict[str, int]],
) -> Optional[GraphSample]:
    try:
        from pycparser import c_parser
    except Exception:
        return None

    try:
        parser = c_parser.CParser()
        ast = parser.parse(source)
    except Exception:
        return None

    ast_edge_map = edge_vocab.get("ast", {})
    child_edge_id = ast_edge_map.get("child", 1)
    next_token_id = ast_edge_map.get("next_token", 2)
    unknown_id = map_unknown(node_vocab)

    node_types: List[int] = []
    node_names: List[str] = []
    ast_edges: List[Tuple[int, int, int]] = []
    traversal_order: List[int] = []

    def map_node(name: str, node, parent_name: Optional[str]) -> str:
        if name == "Decl":
            if parent_name == "ParamList":
                return "ParameterDeclaration"
            if parent_name in {"Compound", "FuncDef"}:
                return "DeclarationStatement"
            return "SimpleDeclaration"
        if name == "IdentifierType":
            names = getattr(node, "names", [])
            if any(item in BUILTIN_TYPES for item in names):
                return "SimpleDeclSpecifier"
            return "NamedTypeSpecifier"
        if name == "TypeDecl":
            return "Declarator"
        if name == "PtrDecl":
            return "Pointer"
        if name == "ArrayDecl":
            return "ArrayDeclarator"
        if name == "ArrayRef":
            return "ArraySubscriptExpression"
        if name == "Struct" or name == "Union" or name == "Enum":
            return "CompositeTypeSpecifier"
        if name == "Typename":
            return "TypeId"
        if name == "Cast":
            return "CastExpression"
        if name == "InitList":
            return "InitializerList"
        if name == "ExprList":
            return "ExpressionList"
        if name == "FuncDef":
            return "FunctionDefinition"
        if name == "FuncDecl":
            return "FunctionDeclarator"
        if name == "ParamList":
            return "ParameterDeclaration"
        if name == "FuncCall":
            return "FunctionCallExpression"
        if name == "BinaryOp":
            return "BinaryExpression"
        if name == "Assignment":
            return "BinaryExpression"
        if name == "UnaryOp":
            return "UnaryExpression"
        if name == "TernaryOp":
            return "ConditionalExpression"
        if name == "ID":
            return "IdExpression"
        if name == "Constant":
            return "LiteralExpression"
        if name == "StructRef":
            return "FieldReference"
        if name == "Return":
            return "ReturnStatement"
        if name == "Break":
            return "BreakStatement"
        if name == "Continue":
            return "ContinueStatement"
        if name == "Goto":
            return "GotoStatement"
        if name == "Label":
            return "LabelStatement"
        if name == "If":
            return "IfStatement"
        if name == "For":
            return "ForStatement"
        if name == "While":
            return "WhileStatement"
        if name == "DoWhile":
            return "DoStatement"
        if name == "Switch":
            return "SwitchStatement"
        if name == "Case":
            return "CaseStatement"
        if name == "Default":
            return "DefaultStatement"
        if name == "Compound":
            return "CompoundStatement"
        if name == "EmptyStatement":
            return "NullStatement"
        if name == "Typedef":
            return "SimpleDeclaration"
        return "ProblemStatement"

    def add_node(vocab_name: str) -> int:
        node_names.append(vocab_name)
        node_types.append(node_vocab.get(vocab_name, unknown_id))
        return len(node_types) - 1

    def walk(node, parent_idx: Optional[int], parent_name: Optional[str]) -> None:
        name = type(node).__name__
        if name == "FileAST":
            for _child_name, child in node.children():
                walk(child, parent_idx=None, parent_name="FileAST")
            return

        vocab_name = map_node(name, node, parent_name)
        idx = add_node(vocab_name)
        if parent_idx is not None:
            ast_edges.append((parent_idx, idx, child_edge_id))
        traversal_order.append(idx)

        for _child_name, child in node.children():
            walk(child, parent_idx=idx, parent_name=name)

    walk(ast, parent_idx=None, parent_name=None)

    for prev_idx, next_idx in zip(traversal_order, traversal_order[1:]):
        ast_edges.append((prev_idx, next_idx, next_token_id))

    cdfg_edges = build_cdfg_edges(node_names, ast_edges, edge_vocab)
    return GraphSample(node_types=node_types, node_names=node_names, ast_edges=ast_edges, cdfg_edges=cdfg_edges)


def tokenize_fallback(
    source: str,
    node_vocab: Dict[str, int],
    edge_vocab: Dict[str, Dict[str, int]],
) -> GraphSample:
    ast_edge_map = edge_vocab.get("ast", {})
    next_token_id = ast_edge_map.get("next_token", 2)
    unknown_id = map_unknown(node_vocab)

    tokens = [match.group(0) for match in TOKEN_PATTERN.finditer(source)]
    node_types: List[int] = []
    node_names: List[str] = []

    for idx, token in enumerate(tokens):
        lowered = token.lower()
        vocab_name = None

        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
            if lowered in KEYWORD_NODE_MAP:
                vocab_name = KEYWORD_NODE_MAP[lowered]
            elif lowered in BUILTIN_TYPES:
                vocab_name = "SimpleDeclSpecifier"
            elif idx + 1 < len(tokens) and tokens[idx + 1] == "(":
                vocab_name = "FunctionCallExpression"
            else:
                vocab_name = "IdExpression"
        elif re.match(r"^(0x[0-9A-Fa-f]+|\d+)$", token) or token.startswith(("\"", "'")):
            vocab_name = "LiteralExpression"
        elif token in {"!", "~", "++", "--"}:
            vocab_name = "UnaryExpression"
        elif re.match(r"^[+\-*/%<>=!&|^]+$", token):
            vocab_name = "BinaryExpression"

        if vocab_name:
            node_names.append(vocab_name)
            node_types.append(node_vocab.get(vocab_name, unknown_id))

    ast_edges: List[Tuple[int, int, int]] = []
    for prev_idx, next_idx in zip(range(len(node_types)), range(1, len(node_types))):
        ast_edges.append((prev_idx, next_idx, next_token_id))

    cdfg_edges = build_cdfg_edges(node_names, ast_edges, edge_vocab)
    return GraphSample(node_types=node_types, node_names=node_names, ast_edges=ast_edges, cdfg_edges=cdfg_edges)


def build_graph_sample(
    source: str,
    node_vocab: Dict[str, int],
    edge_vocab: Dict[str, Dict[str, int]],
) -> GraphSample:
    sanitized = strip_comments(source)
    sample = parse_with_pycparser(sanitized, node_vocab, edge_vocab)
    if sample and sample.node_types:
        return sample
    return tokenize_fallback(sanitized, node_vocab, edge_vocab)


def tensorize(sample: GraphSample, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = len(sample.node_types)
    node_types = torch.tensor([sample.node_types], dtype=torch.long, device=device)
    node_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=device)
    ast_edges = torch.zeros(1, num_nodes, num_nodes, dtype=torch.long, device=device)
    cdfg_edges = torch.zeros(1, num_nodes, num_nodes, dtype=torch.long, device=device)

    for src, dst, edge_type in sample.ast_edges:
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            ast_edges[0, src, dst] = edge_type
    for src, dst, edge_type in sample.cdfg_edges:
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            cdfg_edges[0, src, dst] = edge_type

    return node_types, node_mask, ast_edges, cdfg_edges


def load_model(model_path: Path, device: torch.device) -> Tuple[VulnDetectorGraphTransformer, dict, dict, dict]:
    state, node_vocab, edge_vocab, config = load_checkpoint(model_path, device)
    state_dim = config.get("state_dim", 128)
    num_layers = config.get("num_layers", 4)
    num_heads = config.get("num_heads", 4)
    dropout = config.get("dropout", 0.1)

    model = VulnDetectorGraphTransformer(
        vocab_size=vocab_size(node_vocab),
        ast_edge_types=edge_type_count(edge_vocab.get("ast", {})),
        cdfg_edge_types=edge_type_count(edge_vocab.get("cdfg", {})),
        dim=state_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, node_vocab, edge_vocab, config


def predict_file(
    model: VulnDetectorGraphTransformer,
    path: Path,
    node_vocab: Dict[str, int],
    edge_vocab: Dict[str, Dict[str, int]],
    device: torch.device,
    threshold: float,
) -> Tuple[float, int, int]:
    source = path.read_text(encoding="utf-8", errors="ignore")
    sample = build_graph_sample(source, node_vocab, edge_vocab)
    if not sample.node_types:
        raise ValueError("No nodes extracted from source.")

    node_types, node_mask, ast_edges, cdfg_edges = tensorize(sample, device)
    with torch.no_grad():
        logits = model(node_types, node_mask, ast_edges, cdfg_edges)
        prob = torch.sigmoid(logits).item()
    label = 1 if prob >= threshold else 0
    return prob, label, len(sample.node_types)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_model_dir = script_dir / "Trained_Model"
    default_files = [script_dir / "badExample.c", script_dir / "goodExample.c"]

    parser = argparse.ArgumentParser(description="Detect vulnerable code using a trained Graph Transformer.")
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("files", nargs="*", type=Path, default=default_files)
    args = parser.parse_args()

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model_path = resolve_model_path(args.model_dir, args.model_path)

    model, node_vocab, edge_vocab, config = load_model(model_path, device)
    threshold = args.threshold if args.threshold is not None else float(config.get("threshold", 0.5))

    for file_path in args.files:
        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue
        try:
            prob, label, node_count = predict_file(
                model=model,
                path=file_path,
                node_vocab=node_vocab,
                edge_vocab=edge_vocab,
                device=device,
                threshold=threshold,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to analyze {file_path}: {exc}")
            continue

        verdict = "Vulnerable" if label == 1 else "Not Vulnerable"
        print(f"{file_path.name}: prob={prob:.4f} -> {verdict} (nodes={node_count})")


if __name__ == "__main__":
    main()
