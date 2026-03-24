"""
train.py — Script de treinamento (fine-tuning) do CodeBERT para classificação
           de padrões de projeto em código Python.

Fluxo:
  1. Carrega exemplos do dataset (pastas com arquivos .py)
  2. Tokeniza com o tokenizer do CodeBERT
  3. Divide em treino / validação (80 / 20)
  4. Treina com loop customizado PyTorch (AdamW + CrossEntropyLoss)
  5. Salva modelo e tokenizer ao final

Exemplo de uso:
    python train.py --data_dir dataset --output_dir models/codebert_dp --epochs 3 --batch_size 2
"""

import argparse
import gc
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from preprocessing import Preprocessing, LABELS
from datasets import CodeDataset
from model import CodeBERTClassifier


# ==================== Utilidades ====================

def get_device() -> torch.device:
    """Seleciona o melhor dispositivo disponível: MPS (Apple), CUDA ou CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==================== Loop de treino ====================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Executa uma época de treinamento. Retorna (loss_média, acurácia)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="  Treino", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Avalia o modelo no conjunto de validação. Retorna (loss_média, acurácia)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="  Valid.", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning de CodeBERT para padrões de projeto")
    parser.add_argument("--data_dir", type=str,
                        default="dataset", help="Diretório raiz do dataset")
    parser.add_argument("--output_dir", type=str,
                        default="models/codebert_dp", help="Diretório de saída do modelo")
    parser.add_argument("--model_name", type=str,
                        default="microsoft/codebert-base", help="Nome do modelo HuggingFace")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Número de épocas de treino")
    parser.add_argument("--batch_size", type=int,
                        default=2, help="Tamanho do batch")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Taxa de aprendizado")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Comprimento máximo de tokens")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fração do dataset para validação")
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Se passado, congela os pesos do encoder (treina só a cabeça de classificação)",
    )
    args = parser.parse_args()

    # 1. Dispositivo
    device = get_device()
    print(f"[INFO] Dispositivo: {device}")

    # 2. Tokenizer
    print(f"[INFO] Carregando tokenizer de '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 3. Carregar exemplos
    examples = Preprocessing.load_examples(args.data_dir)
    if len(examples) == 0:
        print(
            "[ERRO] Nenhum exemplo encontrado. Verifique a estrutura de pastas do dataset.")
        return

    # 4. Split treino / validação
    train_examples, val_examples = Preprocessing.split_examples(
        examples, val_ratio=args.val_ratio)
    print(
        f"[INFO] Treino: {len(train_examples)} exemplos | Validação: {len(val_examples)} exemplos")

    # 5. Criar Datasets e DataLoaders
    train_dataset = CodeDataset(
        train_examples, tokenizer, max_length=args.max_length)
    val_dataset = CodeDataset(val_examples, tokenizer,
                              max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    # 6. Modelo
    print(
        f"[INFO] Carregando modelo '{args.model_name}' com {len(LABELS)} classes...")
    model = CodeBERTClassifier(
        model_name=args.model_name,
        num_labels=len(LABELS),
        dropout=0.3,
    ).to(device)

    # 6b. Congelar encoder para economizar memória (treina apenas a cabeça)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("[INFO] Encoder congelado — treinando apenas a cabeça de classificação")
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = model.parameters()

    # Libera memória não utilizada
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 7. Otimizador e função de perda
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 8. Loop de treinamento
    print(f"\n{'='*60}")
    print(f"  Iniciando treinamento — {args.epochs} épocas")
    print(f"{'='*60}\n")

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"Época {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"  Treino  — Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Valid.  — Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # Salva o melhor modelo (baseado em acurácia de validação)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            _save_model(model, tokenizer, args.output_dir)
            print(
                f"  ✓ Modelo salvo em '{args.output_dir}' (melhor val_acc: {best_val_acc:.4f})")

        print()

    print(
        f"Treinamento concluído! Melhor acurácia de validação: {best_val_acc:.4f}")
    print(f"Modelo salvo em: {args.output_dir}")


def _save_model(model, tokenizer, output_dir):
    """Salva state_dict do modelo e tokenizer no diretório de saída."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(
        output_dir, "model_state_dict.pt"))
    tokenizer.save_pretrained(output_dir)
    # Salva também a lista de labels para referência
    with open(os.path.join(output_dir, "labels.txt"), "w") as f:
        for label in LABELS:
            f.write(label + "\n")


if __name__ == "__main__":
    main()
