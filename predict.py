"""
predict.py — Script de inferência para classificar padrões de projeto em
             arquivos Python usando o modelo CodeBERT treinado.

Uso:
    python predict.py --model_dir models/codebert_dp --file meu_codigo.py

O script:
  1. Carrega o tokenizer e o modelo salvo pelo train.py
  2. Lê o arquivo Python informado
  3. Tokeniza o código e faz a predição
  4. Exibe o padrão de projeto previsto e as probabilidades de cada classe
"""

import argparse
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import CodeBERTClassifier
from preprocessing import LABELS, LABEL2ID


def get_device() -> torch.device:
    """Seleciona o melhor dispositivo disponível."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_dir: str, device: torch.device):
    """
    Carrega o tokenizer e o modelo treinado a partir de `model_dir`.
    Retorna (model, tokenizer).
    """
    # Carrega tokenizer salvo
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Carrega labels (opcional — usamos a lista padrão se o arquivo não existir)
    labels_path = os.path.join(model_dir, "labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f if line.strip()]
    else:
        labels = LABELS

    num_labels = len(labels)

    # Instancia o modelo e carrega os pesos treinados
    model = CodeBERTClassifier(num_labels=num_labels)
    state_dict_path = os.path.join(model_dir, "model_state_dict.pt")
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer, labels


@torch.no_grad()
def predict(model, tokenizer, code: str, labels: list, device: torch.device, max_length: int = 512):
    """
    Recebe uma string de código Python e retorna o padrão previsto
    com as probabilidades de cada classe.

    Retorna:
        predicted_label : str   — nome do padrão previsto
        probabilities   : dict  — {label: probabilidade}
    """
    # Tokeniza o código
    encoding = tokenizer(
        code,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inferência
    logits = model(input_ids, attention_mask)  # (1, num_labels)

    # Softmax para obter probabilidades
    probs = F.softmax(logits, dim=1).squeeze(0)  # (num_labels,)

    # Classe com maior probabilidade
    pred_idx = probs.argmax().item()
    predicted_label = labels[pred_idx]

    # Monta dicionário de probabilidades
    probabilities = {labels[i]: probs[i].item() for i in range(len(labels))}

    return predicted_label, probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Predição de padrão de projeto com CodeBERT")
    parser.add_argument("--model_dir", type=str, default="models/codebert_dp",
                        help="Diretório com o modelo treinado")
    parser.add_argument("--file", type=str, required=True,
                        help="Caminho para o arquivo .py a ser classificado")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Comprimento máximo de tokens")
    args = parser.parse_args()

    # 1. Verificar se o arquivo existe
    if not os.path.isfile(args.file):
        print(f"[ERRO] Arquivo não encontrado: {args.file}")
        return

    # 2. Dispositivo
    device = get_device()

    # 3. Carregar modelo e tokenizer
    print(f"[INFO] Carregando modelo de '{args.model_dir}'...")
    model, tokenizer, labels = load_model(args.model_dir, device)

    # 4. Ler o código do arquivo
    with open(args.file, "r", encoding="utf-8") as f:
        code = f.read()

    # 5. Fazer predição
    predicted_label, probabilities = predict(
        model, tokenizer, code, labels, device, args.max_length)

    # 6. Exibir resultado
    print(f"\n{'='*50}")
    print(f"  Arquivo: {args.file}")
    print(f"  Padrão previsto: {predicted_label.upper()}")
    print(f"{'='*50}")
    print("\n  Probabilidades:")
    for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"    {label:<12} {prob:6.2%}  {bar}")
    print()


if __name__ == "__main__":
    main()
