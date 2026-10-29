#!/usr/bin/env python3
"""
Generates model comparison plots comparing ResNet-1D CNN vs GRU vs Scikit-learn baseline.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    m_cnn = pd.read_csv("results/metrics_torch.csv")
    m_gru = pd.read_csv("results/metrics_torch_gru.csv")
    m_skl = pd.read_csv("results/metrics_sklearn.csv")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Sidewalk Test Accuracy Comparison across L (HL=1)
    L_common = [5, 10, 20]
    cnn_sw = m_cnn[(m_cnn.task == "sidewalk") & (m_cnn.hl == 1) & (m_cnn.L.isin(L_common))].sort_values("L")
    gru_sw = m_gru[(m_gru.task == "sidewalk") & (m_gru.hl == 1) & (m_gru.L.isin(L_common))].sort_values("L")
    skl_sw = m_skl[(m_skl.task == "sidewalk") & (m_skl.hl == 1) & (m_skl.L.isin(L_common))].sort_values("L")

    ax1.plot(L_common, cnn_sw.test_acc * 100, "o-", color="#0b7a5a", lw=2, ms=6, label="ResNet-1D CNN (HL1)")
    ax1.plot(L_common, gru_sw.test_acc * 100, "s--", color="#d9534f", lw=2, ms=6, label="GRU Baseline (HL1)")
    ax1.plot(L_common, skl_sw.test_acc * 100, "^:", color="#337ab7", lw=2, ms=6, label="HistGradBoost (HL1)")
    ax1.set_xlabel("Sequence Length (SL, 0.2s slots)")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("(a) Sidewalk Riding Detection: Architecture Comparison (HL 1)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 2. Speeding Test Accuracy Comparison across L (HL=1)
    cnn_sp = m_cnn[(m_cnn.task == "speeding") & (m_cnn.hl == 1) & (m_cnn.L.isin(L_common))].sort_values("L")
    gru_sp = m_gru[(m_gru.task == "speeding") & (m_gru.hl == 1) & (m_gru.L.isin(L_common))].sort_values("L")
    skl_sp = m_skl[(m_skl.task == "speeding") & (m_skl.hl == 1) & (m_skl.L.isin(L_common))].sort_values("L")

    ax2.plot(L_common, cnn_sp.test_acc * 100, "o-", color="#0b7a5a", lw=2, ms=6, label="ResNet-1D CNN (HL1)")
    ax2.plot(L_common, gru_sp.test_acc * 100, "s--", color="#d9534f", lw=2, ms=6, label="GRU Baseline (HL1)")
    ax2.plot(L_common, skl_sp.test_acc * 100, "^:", color="#337ab7", lw=2, ms=6, label="HistGradBoost (HL1)")
    ax2.set_xlabel("Sequence Length (SL, 0.2s slots)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("(b) Speeding Detection: Architecture Comparison (HL 1)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # 3. Latency vs Accuracy Trade-off (Sidewalk, L=10)
    models = ["ResNet CNN HL1", "ResNet CNN HL2", "GRU HL1", "GRU HL2", "Sklearn HL1", "Sklearn HL2"]
    accs = [
        m_cnn[(m_cnn.task == "sidewalk") & (m_cnn.L == 10) & (m_cnn.hl == 1)].test_acc.iloc[0] * 100,
        m_cnn[(m_cnn.task == "sidewalk") & (m_cnn.L == 10) & (m_cnn.hl == 2)].test_acc.iloc[0] * 100,
        m_gru[(m_gru.task == "sidewalk") & (m_gru.L == 10) & (m_gru.hl == 1)].test_acc.iloc[0] * 100,
        m_gru[(m_gru.task == "sidewalk") & (m_gru.L == 10) & (m_gru.hl == 2)].test_acc.iloc[0] * 100,
        m_skl[(m_skl.task == "sidewalk") & (m_skl.L == 10) & (m_skl.hl == 1)].test_acc.iloc[0] * 100,
        m_skl[(m_skl.task == "sidewalk") & (m_skl.L == 10) & (m_skl.hl == 2)].test_acc.iloc[0] * 100,
    ]
    latencies = [
        m_cnn[(m_cnn.task == "sidewalk") & (m_cnn.L == 10) & (m_cnn.hl == 1)].exec_ms.iloc[0],
        m_cnn[(m_cnn.task == "sidewalk") & (m_cnn.L == 10) & (m_cnn.hl == 2)].exec_ms.iloc[0],
        m_gru[(m_gru.task == "sidewalk") & (m_gru.L == 10) & (m_gru.hl == 1)].exec_ms.iloc[0],
        m_gru[(m_gru.task == "sidewalk") & (m_gru.L == 10) & (m_gru.hl == 2)].exec_ms.iloc[0],
        m_skl[(m_skl.task == "sidewalk") & (m_skl.L == 10) & (m_skl.hl == 1)].exec_ms.iloc[0],
        m_skl[(m_skl.task == "sidewalk") & (m_skl.L == 10) & (m_skl.hl == 2)].exec_ms.iloc[0],
    ]
    colors = ["#0b7a5a", "#054d38", "#d9534f", "#a52824", "#337ab7", "#1d5280"]
    markers = ["o", "o", "s", "s", "^", "^"]

    for m, a, l, c, mk in zip(models, accs, latencies, colors, markers):
        ax3.scatter(l, a, color=c, marker=mk, s=90, label=m)
    ax3.set_xlabel("Inference Latency (ms per window)")
    ax3.set_ylabel("Test Accuracy (%)")
    ax3.set_title("(c) Accuracy vs. Inference Latency (Sidewalk, SL 10)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower right", fontsize=8)

    # 4. Model Size (KB)
    sizes = [
        m_cnn[(m_cnn.task == "sidewalk") & (m_cnn.L == 10) & (m_cnn.hl == 1)].size_kb.iloc[0],
        m_cnn[(m_cnn.task == "sidewalk") & (m_cnn.L == 10) & (m_cnn.hl == 2)].size_kb.iloc[0],
        m_gru[(m_gru.task == "sidewalk") & (m_gru.L == 10) & (m_gru.hl == 1)].size_kb.iloc[0],
        m_gru[(m_gru.task == "sidewalk") & (m_gru.L == 10) & (m_gru.hl == 2)].size_kb.iloc[0],
        m_skl[(m_skl.task == "sidewalk") & (m_skl.L == 10) & (m_skl.hl == 1)].size_kb.iloc[0],
        m_skl[(m_skl.task == "sidewalk") & (m_skl.L == 10) & (m_skl.hl == 2)].size_kb.iloc[0],
    ]
    bars = ax4.bar(range(len(models)), sizes, color=colors, edgecolor="k", lw=0.6)
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax4.set_ylabel("Checkpoint Storage Footprint (KB)")
    ax4.set_title("(d) Embedded Flash / Storage Usage by Model (SL 10)")
    ax4.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax4.annotate(f"{h:.1f}K", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig("results/fig_model_comparison.png", dpi=180)
    plt.close()
    print("Saved comprehensive model comparison plot to results/fig_model_comparison.png")

if __name__ == "__main__":
    main()
