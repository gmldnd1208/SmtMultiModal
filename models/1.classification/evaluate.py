<<<<<<< HEAD
"""
평가 스크립트 (evaluate.py)
=============================
목적:
  학습된 SMTCauseAnalyzer 모델을 검증/테스트 데이터로 평가하고
  (1) 불량 유형별 정밀도·재현율·F1 리포트
  (2) 불량 유형별 센서 기여도 평균 (원인 분석)
  (3) 전체 요약 출력 및 CSV 저장

실행:
  cd models/1.classification

  # 기본: processed_data/val 데이터로 best_model.pth 평가
  python evaluate.py --checkpoint results/<exp_id>/checkpoints/best_model.pth

  # 특정 CSV 지정 (train 데이터 평가 등)
  python evaluate.py --checkpoint results/<exp_id>/checkpoints/best_model.pth \\
                     --csv processed_data/train/metadata.csv

출력:
  results/<exp_id>/
    evaluation_report.csv   ← 불량 유형별 Precision / Recall / F1 / 센서 기여도
    evaluation_summary.txt  ← 전체 지표 요약 텍스트
"""

import argparse
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import SMTCauseAnalyzer, DEFECT_NAMES, SENSOR_NAMES

IMG_SIZE    = 224    # 학습 시와 동일한 크기
BATCH_SIZE  = 32   # 한 번에 추론할 이미지 수 (GPU 메모리 부족 시 16으로 줄일 것)
NUM_WORKERS = 0    # Windows 환경 DataLoader worker crash 방지 (train.py와 동일)
THRESHOLD   = 0.5   # 모델 출력 확률이 이 값 이상이면 해당 불량이 "있다"고 판정

LABEL_COLS = [f"label_{n}" for n in DEFECT_NAMES]


def find_best_checkpoint() -> Path | None:
    """results/ 내 타임스탬프 폴더를 정렬해 가장 최신 best_model.pth를 반환한다."""
    candidates = sorted(Path("results").glob("*/checkpoints/best_model.pth"))
    return candidates[-1] if candidates else None


class SMTEvalDataset(Dataset):
    """
    평가 전용 데이터셋.
    학습 시와 동일한 정규화를 적용하되, 데이터 증강은 사용하지 않는다.
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row    = self.df.iloc[idx]
        image  = np.array(Image.open(row["image_path"]).convert("RGB"))
        image  = self.transform(image=image)["image"]            # (3, H, W)
        sensor = torch.FloatTensor(np.load(row["sensor_path"]))  # (5, 5)
        label  = torch.FloatTensor(row[LABEL_COLS].values.astype(float))  # (16,)
        return image, sensor, label


@torch.no_grad()
def run_inference(model, loader, device):
    """
    모든 배치에 대해 추론을 수행하고 결과를 수집한다.

    Returns:
        all_probs  : (N, 16)  — sigmoid 확률
        all_attns  : (N, 5)   — 센서 기여도 (softmax)
        all_labels : (N, 16)  — 정답 라벨
    """
    model.eval()  # Dropout 비활성화, BatchNorm 고정 통계 사용 (추론 모드)
    all_probs, all_attns, all_labels = [], [], []  # 배치별 결과를 리스트로 모아 나중에 합산

    for images, sensors, labels in tqdm(loader, desc="  추론 중", leave=False):
        # 모델과 같은 장치(GPU/CPU)로 데이터 이동
        images  = images.to(device)
        sensors = sensors.to(device)
        # labels는 지표 계산용이므로 GPU로 옮기지 않고 CPU에 유지

        # AMP: CUDA 환경에서는 FP16 연산으로 추론 속도 향상, CPU에서는 일반 FP32 사용
        amp_ctx = torch.amp.autocast("cuda") if torch.cuda.is_available() else contextlib.nullcontext()
        with amp_ctx:
            logits, attn = model(images, sensors)

        # logit(raw 점수)을 0~1 사이 확률로 변환 (sigmoid)
        probs = torch.sigmoid(logits).cpu()   # (B, 16) — 불량 유형별 확률
        attn  = attn.cpu()                    # (B, 5)  — 센서별 기여도

        # 배치 결과를 리스트에 누적 (전체 데이터 처리 후 torch.cat으로 합침)
        all_probs.append(probs)
        all_attns.append(attn)
        all_labels.append(labels)

    return (
        torch.cat(all_probs),   # (N, 16)
        torch.cat(all_attns),   # (N, 5)
        torch.cat(all_labels),  # (N, 16)
    )


def compute_per_class_metrics(probs, labels, threshold=THRESHOLD):
    """
    불량 유형별 Precision / Recall / F1 / Support 를 계산한다.

    Args:
        probs : (N, 16) — sigmoid 확률
        labels: (N, 16) — 정답 (0 or 1)

    Returns:
        DataFrame: 각 행이 하나의 불량 유형, 열은 precision/recall/f1/support
    """
    preds = (probs >= threshold).float()   # (N, 16) 이진 예측

    rows = []
    for i, name in enumerate(DEFECT_NAMES):
        tp = (preds[:, i] * labels[:, i]).sum().item()
        fp = (preds[:, i] * (1 - labels[:, i])).sum().item()
        fn = ((1 - preds[:, i]) * labels[:, i]).sum().item()
        tn = ((1 - preds[:, i]) * (1 - labels[:, i])).sum().item()

        support   = int(labels[:, i].sum().item())   # 데이터셋에 실제로 존재하는 해당 불량 건수
        precision = tp / (tp + fp + 1e-8)   # 불량이라 예측한 것 중 실제 불량 비율 (정확도)
        recall    = tp / (tp + fn + 1e-8)   # 실제 불량 중 맞게 예측한 비율 (검출률)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)  # precision·recall 조화 평균
        accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-8)  # 전체 샘플 중 정답 비율

        rows.append({
            "불량_유형":   name,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "accuracy":  round(accuracy, 4),
            "support":   support,           # 실제 불량 건수
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
        })

    return pd.DataFrame(rows)


def compute_sensor_contrib_per_defect(probs, attns, threshold=THRESHOLD):
    """
    각 불량 유형이 감지된 샘플들의 평균 센서 기여도를 계산한다.

    해석 방법:
      - 특정 불량 유형이 predicted=1 인 샘플들만 선택
      - 그 샘플들에서 센서 기여도(attention)를 평균
      → "이 불량이 발생했을 때 어떤 센서가 주로 이상했는가"

    Args:
        probs : (N, 16) — sigmoid 확률
        attns : (N, 5)  — 센서 기여도

    Returns:
        DataFrame: 각 행이 불량 유형, 열이 센서 이름 (기여도 평균값)
    """
    preds = (probs >= threshold).float()  # (N, 16) 이진 예측

    rows = []
    for i, defect_name in enumerate(DEFECT_NAMES):
        # 해당 불량을 예측한 샘플 인덱스 (True Positive + False Positive 모두 포함)
        predicted_mask = preds[:, i].bool()

        if predicted_mask.sum() == 0:
            # 해당 불량이 한 번도 예측되지 않은 경우
            avg_contrib = {s: 0.0 for s in SENSOR_NAMES}
        else:
            # 예측된 샘플들의 센서 기여도 평균
            selected_attns = attns[predicted_mask]         # (M, 5)
            avg            = selected_attns.mean(dim=0)    # (5,)
            avg_contrib    = {s: round(avg[j].item(), 4) for j, s in enumerate(SENSOR_NAMES)}

        rows.append({"불량_유형": defect_name, **avg_contrib})

    return pd.DataFrame(rows)


def compute_overall_metrics(probs, labels, threshold=THRESHOLD):
    """
    전체 데이터셋의 요약 지표를 계산한다.

    Returns:
        dict: exact_match, macro_f1, macro_precision, macro_recall, 샘플수
    """
    preds = (probs >= threshold).float()

    # Exact Match: 16개 불량 유형을 하나도 빠짐없이 전부 맞춘 샘플 비율
    # 매우 엄격한 지표 — 16개 중 하나라도 틀리면 0점
    exact_match = (preds == labels).all(dim=1).float().mean().item()

    # 16개 클래스 각각에 대해 TP/FP/FN 집계 (dim=0: 샘플 방향으로 합산)
    # TP(True Positive) : 불량이라 예측했고 실제로도 불량
    # FP(False Positive): 불량이라 예측했지만 실제는 정상 (오탐)
    # FN(False Negative): 정상이라 예측했지만 실제는 불량 (미탐)
    tp = (preds * labels).sum(dim=0)           # (16,)
    fp = (preds * (1 - labels)).sum(dim=0)     # (16,)
    fn = ((1 - preds) * labels).sum(dim=0)     # (16,)

    # 16개 클래스 각각의 precision/recall/f1 계산 후 평균 (Macro 방식)
    # 1e-8: 해당 불량이 데이터에 없어 TP=FP=FN=0일 때 0나누기 방지
    precision_per = tp / (tp + fp + 1e-8)   # (16,)
    recall_per    = tp / (tp + fn + 1e-8)   # (16,)
    f1_per        = 2 * precision_per * recall_per / (precision_per + recall_per + 1e-8)
    precision     = precision_per.mean().item()
    recall        = recall_per.mean().item()
    macro_f1      = f1_per.mean().item()    # 16개 클래스 F1의 단순 평균

    return {
        "total_samples":    int(probs.shape[0]),
        "exact_match":      round(exact_match, 4),
        "macro_f1":         round(macro_f1, 4),
        "macro_precision":  round(precision, 4),
        "macro_recall":     round(recall, 4),
    }


def print_report(overall, per_class_df, sensor_df):
    """콘솔에 평가 결과를 보기 좋게 출력한다."""

    print("\n" + "=" * 60)
    print("  전체 요약")
    print("=" * 60)
    print(f"  샘플 수        : {overall['total_samples']:,}")
    print(f"  Exact Match    : {overall['exact_match']:.4f}")
    print(f"  Macro F1       : {overall['macro_f1']:.4f}")
    print(f"  Macro Precision: {overall['macro_precision']:.4f}")
    print(f"  Macro Recall   : {overall['macro_recall']:.4f}")

    print("\n" + "=" * 60)
    print("  불량 유형별 성능 (F1 내림차순)")
    print("=" * 60)
    sorted_df = per_class_df.sort_values("f1", ascending=False)
    print(f"  {'불량 유형':<14} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>8}")
    print("  " + "-" * 52)
    for _, row in sorted_df.iterrows():
        print(f"  {row['불량_유형']:<14} {row['precision']:>10.4f} "
              f"{row['recall']:>10.4f} {row['f1']:>8.4f} {row['support']:>8}")

    print("\n" + "=" * 60)
    print("  불량 유형별 주요 원인 센서 (기여도 평균)")
    print("=" * 60)
    print(f"  {'불량 유형':<14}", end="")
    for s in SENSOR_NAMES:
        print(f" {s:>12}", end="")
    print()
    print("  " + "-" * (14 + 13 * len(SENSOR_NAMES)))
    for _, row in sensor_df.iterrows():
        print(f"  {row['불량_유형']:<14}", end="")
        # 기여도가 가장 높은 센서를 강조 표시 (*)
        values  = [row[s] for s in SENSOR_NAMES]
        max_idx = values.index(max(values))
        for j, v in enumerate(values):
            marker = "*" if j == max_idx else " "
            print(f" {v:>11.4f}{marker}", end="")
        print()
    print("  (* 표시: 해당 불량의 주요 원인 센서)")


def main(checkpoint: str | None, csv_path: str):
    # 체크포인트 경로 결정 (미지정 시 최신 자동 탐색)
    if checkpoint:
        ckpt_path = Path(checkpoint)
    else:
        ckpt_path = find_best_checkpoint()
        if ckpt_path is None:
            print("[오류] results/ 폴더에서 best_model.pth를 찾을 수 없습니다. 먼저 train.py를 실행하거나 --checkpoint로 경로를 지정하세요.")
            return

    if not ckpt_path.exists():
        print(f"[오류] 체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
        return

    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"[오류] CSV 파일을 찾을 수 없습니다: {csv_file}")
        return

    # 결과를 체크포인트와 같은 실험 폴더에 저장
    # ex: results/20250623_143022/checkpoints/best_model.pth
    #   → results/20250623_143022/
    out_dir = ckpt_path.parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"평가 장치: {device}")
    print(f"체크포인트: {ckpt_path}")
    print(f"데이터 CSV: {csv_file}")

    # 모델 로드
    model = SMTCauseAnalyzer().to(device)

    # best_model.pth: 모델 state_dict 만 저장
    # last.pth      : epoch/optimizer 등 포함된 전체 dict
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict) and "model" in raw:
        # last.pth 형식
        model.load_state_dict(raw["model"])
        saved_epoch = raw.get("epoch", "?")
        print(f"last.pth 로드 (epoch={saved_epoch})")
    else:
        # best_model.pth 형식 (state_dict 만)
        model.load_state_dict(raw)
        print("best_model.pth 로드")

    # DataLoader 설정
    loader = DataLoader(
        SMTEvalDataset(csv_file),
        batch_size=BATCH_SIZE,
        shuffle=False,              # 평가는 순서 고정
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    # 추론
    print(f"\n총 {len(loader.dataset):,}개 샘플 추론 시작...")
    probs, attns, labels = run_inference(model, loader, device)

    # 지표 계산
    overall       = compute_overall_metrics(probs, labels)
    per_class_df  = compute_per_class_metrics(probs, labels)
    sensor_df     = compute_sensor_contrib_per_defect(probs, attns)

    # 콘솔 출력
    print_report(overall, per_class_df, sensor_df)

    # 결과 CSV 저장
    # 불량 유형별 성능 + 센서 기여도를 하나의 리포트로 합산
    report_df = per_class_df.merge(sensor_df, on="불량_유형")
    report_path = out_dir / "evaluation_report.csv"
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\n[저장] 리포트 CSV: {report_path}")

    # 전체 요약 텍스트 저장
    summary_path = out_dir / "evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== SMT 불량 원인 분석 평가 요약 ===\n\n")
        f.write(f"체크포인트  : {ckpt_path}\n")
        f.write(f"데이터 CSV  : {csv_file}\n")
        f.write(f"샘플 수     : {overall['total_samples']:,}\n")
        f.write(f"Exact Match : {overall['exact_match']:.4f}\n")
        f.write(f"Macro F1    : {overall['macro_f1']:.4f}\n")
        f.write(f"Macro Prec  : {overall['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {overall['macro_recall']:.4f}\n\n")

        f.write("=== 불량 유형별 주요 원인 센서 ===\n\n")
        for _, row in sensor_df.iterrows():
            values  = {s: row[s] for s in SENSOR_NAMES}
            top_sensor = max(values, key=values.get)
            f.write(f"  {row['불량_유형']:12s} → 주요 원인: {top_sensor} "
                    f"(기여도 {values[top_sensor]:.4f})\n")

    print(f"[저장] 요약 텍스트: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMT 불량 원인 분석 모델 평가")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="평가할 모델 체크포인트 경로 (기본값: results/ 내 최신 best_model.pth 자동 탐색)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="processed_data/val/metadata.csv",
        help="평가에 사용할 metadata.csv 경로 (기본값: val 데이터)",
    )
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, csv_path=args.csv)
=======
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from pathlib import Path

# train.py에서 데이터셋과 모델 구조를 가져옵니다.
try:
    from train import DualEncodingModel, SMTMultimodalDataset
except ImportError:
    print("🚨 에러: train.py 파일에서 모델과 데이터셋 클래스를 불러올 수 없습니다.")
    raise

def plot_confusion_matrix(cm, classes, save_path, title='Confusion Matrix'):
    """혼동 행렬 시각화 및 저장"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """ROC Curve 시각화 및 저장"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (과탐률)')
    plt.ylabel('True Positive Rate (정탐률)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_model(model_path, test_csv, results_dir, device):
    print(f"🔍 [1/4] 평가 준비 중... (디바이스: {device})")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 모델 및 데이터 로드
    model = DualEncodingModel(num_classes=2).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚨 가중치 파일을 찾을 수 없습니다: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    test_dataset = SMTMultimodalDataset(test_csv, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"📊 테스트 데이터: 총 {len(test_dataset)}건")

    # 2. 결과 수집용 컨테이너
    all_preds, all_labels, all_probs = [], [], []
    all_process_types = []
    error_cases = [] # 오답 노트용

    print("🚀 [2/4] 테스트 데이터 추론 시작...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = batch['image'].to(device)
            sensor_data = batch['sensor_data'].to(device)
            labels = batch['label'].to(device)
            process_types = batch['process_type']
            
            outputs = model(images, sensor_data)
            
            # [핵심] Softmax를 통해 0~1 사이의 확률값 추출
            probs = F.softmax(outputs, dim=1)[:, 1] 
            # 2. [수정됨] 커스텀 임계값 설정 및 비교
            custom_threshold = 0.8788
            preds = (probs >= custom_threshold).long()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_process_types.extend(process_types)
            
            # 오답 데이터(FP, FN) 정보 저장
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                
                if true_label != pred_label:
                    # Dataset에서 원본 행을 추적하여 경로 추출
                    global_idx = batch_idx * test_loader.batch_size + i
                    orig_row = test_dataset.df.iloc[global_idx]
                    
                    error_type = "과탐(FP)" if pred_label == 1 else "미탐(FN)"
                    
                    error_cases.append({
                        'Error_Type': error_type,
                        'Process': process_types[i],
                        'True_Label': 'Defect' if true_label == 1 else 'Normal',
                        'Predicted_Label': 'Defect' if pred_label == 1 else 'Normal',
                        'Defect_Probability': f"{probs[i].item()*100:.2f}%",
                        'Image_Path': orig_row['image_path'],
                        'Sensor_Path': orig_row['sensor_path']
                    })

    print("📈 [3/4] 평가지표 계산 및 리포트 생성 중...")
    
    # 3. 평가지표 계산
    class_names = ['Normal(0)', 'Defect(1)']
    cm = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=class_names)
    
    # ROC 커브 및 최적의 Threshold 계산
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Youden's J statistic을 이용한 최적의 임계값 찾기
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # 공정별 분리 분석
    df_results = pd.DataFrame({'Label': all_labels, 'Pred': all_preds, 'Process': all_process_types})
    process_reports = {}
    for p_type in df_results['Process'].unique():
        p_data = df_results[df_results['Process'] == p_type]
        if len(p_data) > 0:
            process_reports[p_type] = classification_report(p_data['Label'], p_data['Pred'], output_dict=True, zero_division=0)

    print("💾 [4/4] 결과물 파일 저장 중...")
    
    # [결과물 1] Confusion Matrix 이미지 저장
    plot_confusion_matrix(cm, class_names, os.path.join(results_dir, 'confusion_matrix.png'))
    
    # [결과물 2] ROC Curve 이미지 저장
    plot_roc_curve(fpr, tpr, roc_auc, os.path.join(results_dir, 'roc_curve.png'))
    
    # [결과물 3] 오답 노트 CSV 저장
    if error_cases:
        df_errors = pd.DataFrame(error_cases)
        # 치명적인 미탐(FN)이 위로 오도록 정렬
        df_errors = df_errors.sort_values(by='Error_Type', ascending=False)
        df_errors.to_csv(os.path.join(results_dir, 'misclassified_cases.csv'), index=False, encoding='utf-8-sig')
    
    # [결과물 4] 종합 평가 리포트 TXT 저장
    with open(os.path.join(results_dir, 'evaluation_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=== SMT 멀티모달 모델 종합 평가 리포트 ===\n")
        f.write(f"테스트 데이터 수: {len(all_labels)}건\n\n")
        
        f.write("1. 전체 분류 성능 (Classification Report)\n")
        f.write(report_str + "\n\n")
        
        f.write("2. 혼동 행렬 상세 (Confusion Matrix)\n")
        f.write(f"- 정상 데이터 정확히 맞춤 (TN): {cm[0][0]}건\n")
        f.write(f"- 정상인데 불량으로 판정 [과탐/FP]: {cm[0][1]}건 (주의 요망)\n")
        f.write(f"- 불량인데 정상으로 판정 [미탐/FN]: {cm[1][0]}건 (★치명적 결함)\n")
        f.write(f"- 불량 데이터 정확히 맞춤 (TP): {cm[1][1]}건\n\n")
        
        f.write("3. 확률 및 임계값 분석\n")
        f.write(f"- ROC-AUC Score: {roc_auc:.4f}\n")
        f.write(f"- 추천 최적 임계값 (Threshold): {optimal_threshold:.4f} (이 값 이상이면 불량으로 판정 시 효율 극대화)\n\n")
        
        f.write("4. 공정별 F1-Score (불량 검출 기준)\n")
        for p_type, p_report in process_reports.items():
            # 라벨 1(불량)이 데이터셋에 존재하는 경우에만 기록
            defect_f1 = p_report.get('1', {}).get('f1-score', 'N/A')
            f.write(f"- {p_type}: {defect_f1}\n")

    print(f"\n🎉 모든 평가가 완료되었습니다. 결과물은 [{results_dir}] 폴더를 확인하세요.")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    
    # test.csv 경로 (preprocess.py에서 생성된 test 메타데이터)
    test_csv_path = current_dir / "processed_data" / "test" / "metadata.csv"
    
    # 저장된 최고의 모델 가중치 경로
    # 모델 폴더를 명시적으로 찾도록 경로 설정
    model_weight_path = current_dir / "results_train" 
    
    # 가장 최근에 생성된 experiment_id 폴더를 찾아서 best_model.pth 선택
    if model_weight_path.exists():
        subdirs = [d for d in model_weight_path.iterdir() if d.is_dir()]
        if subdirs:
            latest_exp_dir = max(subdirs, key=os.path.getmtime)
            best_model_path = latest_exp_dir / "checkpoints" / "best_model.pth"
        else:
            best_model_path = model_weight_path / "best_model.pth" # 기본 백백 경로
    else:
        best_model_path = current_dir / "best_model.pth"

    # 평가 결과물이 저장될 폴더
    eval_results_dir = current_dir / "evaluation_results"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not test_csv_path.exists():
        print(f"🚨 에러: 테스트 데이터 목차({test_csv_path})가 없습니다.")
    else:
        evaluate_model(str(best_model_path), str(test_csv_path), str(eval_results_dir), device)
>>>>>>> 5cf672cc77980edfb9c355066efc67015254ca34
