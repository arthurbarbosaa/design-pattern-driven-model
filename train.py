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

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from algorithms import FineTuningAlgorithm
from observers import ModelCheckpointObserver, TrainingPrintObserver
from preprocessing import Preprocessing, LABELS
from datasets import CodeDataset
from model import CodeBERTClassifier

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
    parser.add_argument("--lr_encoder", type=float, default=1e-5,
                        help="Taxa de aprendizado do encoder")
    parser.add_argument("--lr_classifier", type=float, default=1e-3,
                        help="Taxa de aprendizado da cabeça de classificação")
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
    device = torch.device("cpu")
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

    # 7. Algoritmo de treinamento
    training_algorithm = FineTuningAlgorithm(
        model,
        train_loader,
        val_loader,
        device,
        lr_encoder=args.lr_encoder,
        lr_classifier=args.lr_classifier,
        freeze_encoder=args.freeze_encoder,
    )

    # Observador responsável por imprimir o progresso do treinamento
    training_algorithm.add(TrainingPrintObserver())

    # Observador responsável por salvar o melhor modelo durante o treinamento
    training_algorithm.add(ModelCheckpointObserver(args.output_dir, tokenizer))

    # 8. Loop de treinamento
    training_algorithm.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
