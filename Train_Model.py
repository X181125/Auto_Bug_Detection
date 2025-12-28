import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path, max_samples=None):
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if "num_nodes" not in sample:
                sample["num_nodes"] = len(sample.get("node_types", []))
            samples.append(sample)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def create_batches(samples, batch_size, max_nodes_per_batch, shuffle):
    indices = list(range(len(samples)))
    if shuffle:
        random.shuffle(indices)

    if not max_nodes_per_batch:
        return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]

    indices.sort(key=lambda idx: samples[idx]["num_nodes"])
    batches = []
    current = []
    current_nodes = 0
    for idx in indices:
        nodes = samples[idx]["num_nodes"]
        if nodes == 0:
            continue
        if current and (
            (batch_size and len(current) >= batch_size) or (current_nodes + nodes > max_nodes_per_batch)
        ):
            batches.append(current)
            current = []
            current_nodes = 0
        current.append(idx)
        current_nodes += nodes
    if current:
        batches.append(current)
    if shuffle:
        random.shuffle(batches)
    return batches


def collate(samples, device):
    max_nodes = max(sample["num_nodes"] for sample in samples)
    batch_size = len(samples)
    node_types = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    ast_edges = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.long)
    cdfg_edges = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.long)
    labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.float32)

    for idx, sample in enumerate(samples):
        num_nodes = sample["num_nodes"]
        node_types[idx, :num_nodes] = torch.tensor(sample["node_types"], dtype=torch.long)
        node_mask[idx, :num_nodes] = True

        if sample["ast_edges"]:
            edges = torch.tensor(sample["ast_edges"], dtype=torch.long)
            ast_edges[idx, edges[:, 0], edges[:, 1]] = edges[:, 2]

        if sample["cdfg_edges"]:
            edges = torch.tensor(sample["cdfg_edges"], dtype=torch.long)
            cdfg_edges[idx, edges[:, 0], edges[:, 1]] = edges[:, 2]

    return (
        node_types.to(device),
        node_mask.to(device),
        ast_edges.to(device),
        cdfg_edges.to(device),
        labels.to(device),
    )


class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, edge_type_count, dropout):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.edge_bias = nn.Embedding(edge_type_count + 1, num_heads, padding_idx=0)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_types, node_mask):
        batch_size, num_nodes, _ = x.shape
        residual = x
        x = self.norm1(x)

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        edge_bias = self.edge_bias(edge_types)
        edge_bias = edge_bias.permute(0, 3, 1, 2)
        attn_scores = attn_scores + edge_bias

        edge_mask = edge_types != 0
        eye = torch.eye(num_nodes, device=edge_types.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = edge_mask | eye
        node_mask_2d = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        attn_mask = edge_mask & node_mask_2d

        attn_scores = attn_scores.masked_fill(~attn_mask.unsqueeze(1), -1e9)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.dim)
        out = self.proj(out)
        out = self.dropout(out)
        out = out * node_mask.unsqueeze(-1)
        x = residual + out

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * node_mask.unsqueeze(-1)
        return residual + x


class GraphTransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads, edge_type_count, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(dim, num_heads, edge_type_count, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, edge_types, node_mask):
        for layer in self.layers:
            x = layer(x, edge_types, node_mask)
        return x


class WeightedSumReadout(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x, node_mask):
        scores = self.att(x).squeeze(-1)
        scores = scores.masked_fill(~node_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights.unsqueeze(-1) * x, dim=1)


class VulnDetectorGraphTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        ast_edge_types,
        cdfg_edge_types,
        dim,
        num_layers,
        num_heads,
        dropout,
    ):
        super().__init__()
        self.node_embed = nn.Embedding(vocab_size + 1, dim, padding_idx=0)
        self.ast_encoder = GraphTransformerEncoder(dim, num_layers, num_heads, ast_edge_types, dropout)
        self.cdfg_encoder = GraphTransformerEncoder(dim, num_layers, num_heads, cdfg_edge_types, dropout)
        self.ast_readout = WeightedSumReadout(dim)
        self.cdfg_readout = WeightedSumReadout(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, node_types, node_mask, ast_edge_types, cdfg_edge_types):
        node_embeddings = self.node_embed(node_types)
        ast_out = self.ast_encoder(node_embeddings, ast_edge_types, node_mask)
        cdfg_out = self.cdfg_encoder(node_embeddings, cdfg_edge_types, node_mask)
        ast_graph = self.ast_readout(ast_out, node_mask)
        cdfg_graph = self.cdfg_readout(cdfg_out, node_mask)
        combined = torch.cat([ast_graph, cdfg_graph], dim=-1)
        return self.classifier(combined).squeeze(-1)


def compute_metrics(labels, probs, threshold):
    preds = (probs >= threshold).long()
    labels = labels.long()
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def evaluate(model, samples, batch_size, max_nodes_per_batch, device, threshold):
    model.eval()
    batches = create_batches(samples, batch_size, max_nodes_per_batch, shuffle=False)
    all_labels = []
    all_probs = []
    total_loss = 0.0
    total_count = 0

    for batch in batches:
        batch_samples = [samples[idx] for idx in batch]
        node_types, node_mask, ast_edges, cdfg_edges, labels = collate(batch_samples, device)
        logits = model(node_types, node_mask, ast_edges, cdfg_edges)
        probs = torch.sigmoid(logits)
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())
        total_count += labels.numel()
        total_loss += torch.sum(
            nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        ).item()

    if total_count == 0:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    labels = torch.cat(all_labels)
    probs = torch.cat(all_probs)
    metrics = compute_metrics(labels, probs, threshold)
    metrics["loss"] = total_loss / total_count
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Graph Transformer for Auto Bug Detection.")
    parser.add_argument(
        "--data-dir",
        default=r"C:\Users\HungNguyenDinh\Documents\Visual_Studio_Code_Workspace\Auto_Bug_Detection\TIFS_Data\processed\github_after",
        help="Directory with train.jsonl/val.jsonl/test.jsonl and vocab files.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-nodes-per-batch", type=int, default=8000)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--save-dir",
        default=r"C:\Users\HungNguyenDinh\Documents\Visual_Studio_Code_Workspace\Auto_Bug_Detection\Trained_Model",
        help="Directory to save model checkpoints.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    node_vocab_path = data_dir / "node_vocab.json"
    edge_vocab_path = data_dir / "edge_vocab.json"
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"

    if not train_path.exists():
        raise SystemExit(f"Missing train.jsonl at {train_path}")

    with node_vocab_path.open("r", encoding="utf-8") as handle:
        node_vocab = json.load(handle)

    with edge_vocab_path.open("r", encoding="utf-8") as handle:
        edge_vocab = json.load(handle)

    train_samples = load_jsonl(train_path, args.max_train_samples or None)
    val_samples = load_jsonl(val_path, args.max_val_samples or None) if val_path.exists() else []
    test_samples = load_jsonl(test_path, args.max_test_samples or None) if test_path.exists() else []

    if not train_samples:
        raise SystemExit("No training samples loaded.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    model = VulnDetectorGraphTransformer(
        vocab_size=len(node_vocab),
        ast_edge_types=len(edge_vocab["ast"]),
        cdfg_edge_types=len(edge_vocab["cdfg"]),
        dim=args.state_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    labels = torch.tensor([sample["label"] for sample in train_samples], dtype=torch.float32)
    pos_count = labels.sum().item()
    neg_count = len(labels) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    patience_left = args.patience
    best_path = save_dir / "Auto_Bug_Detector_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = create_batches(
            train_samples,
            batch_size=args.batch_size,
            max_nodes_per_batch=args.max_nodes_per_batch,
            shuffle=True,
        )
        total_loss = 0.0
        total_count = 0

        for batch in batches:
            batch_samples = [train_samples[idx] for idx in batch]
            node_types, node_mask, ast_edges, cdfg_edges, batch_labels = collate(batch_samples, device)

            optimizer.zero_grad()
            logits = model(node_types, node_mask, ast_edges, cdfg_edges)
            loss = criterion(logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * batch_labels.numel()
            total_count += batch_labels.numel()

        train_loss = total_loss / max(total_count, 1)
        val_metrics = (
            evaluate(model, val_samples, args.batch_size, args.max_nodes_per_batch, device, args.threshold)
            if val_samples
            else {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_left = args.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "node_vocab": node_vocab,
                    "edge_vocab": edge_vocab,
                    "config": vars(args),
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state"])

    final_path = save_dir / "Auto_Bug_Detector.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "node_vocab": node_vocab,
            "edge_vocab": edge_vocab,
            "config": vars(args),
        },
        final_path,
    )

    if test_samples:
        test_metrics = evaluate(
            model, test_samples, args.batch_size, args.max_nodes_per_batch, device, args.threshold
        )
        print(
            "Test metrics: "
            f"loss={test_metrics['loss']:.4f} "
            f"acc={test_metrics['accuracy']:.4f} "
            f"f1={test_metrics['f1']:.4f}"
        )

    print(f"Saved model to {final_path}")


if __name__ == "__main__":
    main()
