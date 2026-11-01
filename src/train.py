import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from train_config import TRAIN_CONFIG, set_seed
from src.model import build_model, build_gru
from src.train_utils import load_split, to_ds, get_class_weights

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

def train_one(L, hl, model_class="cnn", save_dir=None, epochs=None, batch_size=None):
    """
    Train a single model configuration (L_seq, hl).
    Implements Task 3.1 with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
    """
    set_seed(TRAIN_CONFIG["seed"])
    if save_dir is None:
        save_dir = TRAIN_CONFIG["paths"].get("sweep", TRAIN_CONFIG["paths"]["models"] / "sweep")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = epochs or TRAIN_CONFIG["epochs"]
    batch_size = batch_size or 64  # Optimal for vectorized CPU throughput

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_split(L)

    train_ds = to_ds(X_train, y_train, batch_size=batch_size, train=True)
    val_ds = to_ds(X_val, y_val, batch_size=batch_size, train=False)
    test_ds = to_ds(X_test, y_test, batch_size=batch_size, train=False)

    class_weight = get_class_weights(y_train)

    if model_class == "cnn":
        model = build_model(
            L_seq=L,
            hl=hl,
            filters=TRAIN_CONFIG["filters"],
            num_classes=TRAIN_CONFIG["num_classes"],
            dropout=TRAIN_CONFIG["dropout"],
            l2=TRAIN_CONFIG["l2"],
        )
    else:
        model = build_gru(
            L_seq=L,
            n_layers=hl,
            units=TRAIN_CONFIG["filters"],
            num_classes=TRAIN_CONFIG["num_classes"],
            dropout=TRAIN_CONFIG["dropout"],
            l2=TRAIN_CONFIG["l2"],
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TRAIN_CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    h5_path = save_dir / f"{model_class}_L{L}_hl{hl}.h5"
    model_path = h5_path
    confusion_dir = TRAIN_CONFIG["paths"]["confusion"]
    cm_csv_path = confusion_dir / f"{model_class}_L{L}_hl{hl}.csv"

    # Check if this model is already trained and evaluated
    keras_path = save_dir / f"{model_class}_L{L}_hl{hl}.keras"
    if (h5_path.exists() or keras_path.exists()) and cm_csv_path.exists():
        load_target = h5_path if h5_path.exists() else keras_path
        print(f"\n[Resume] Found trained model {load_target.name}. Evaluating...")
        try:
            model = tf.keras.models.load_model(str(load_target))
            val_metrics = model.evaluate(val_ds, verbose=0)
            val_loss, val_acc, val_prec, val_rec = val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3]
            test_metrics = model.evaluate(test_ds, verbose=0)
            test_loss, test_acc, test_prec, test_rec = test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3]

            cm_df = pd.read_csv(cm_csv_path, index_col=0)
            cm_norm = cm_df.values

            n_params = model.count_params()
            size_file = model_path if model_path.exists() else h5_path
            size_kb = os.path.getsize(size_file) / 1024.0 if size_file.exists() else 0.0

            record = {
                "model": model_class,
                "L_seq": L,
                "hl": hl,
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "val_prec": float(val_prec),
                "val_rec": float(val_rec),
                "test_acc": float(test_acc),
                "test_loss": float(test_loss),
                "test_prec": float(test_prec),
                "test_rec": float(test_rec),
                "tn_rate": float(cm_norm[0, 0]),
                "fp_rate": float(cm_norm[0, 1]),
                "fn_rate": float(cm_norm[1, 0]),
                "tp_rate": float(cm_norm[1, 1]),
                "params": int(n_params),
                "size_kb": float(size_kb),
                "epoch_best": 0,
                "epochs_trained": 0,
                "wall_time_s": 0.0,
            }
            print(f"Loaded {model_class.upper()} L={L} HL={hl}: Val Acc={val_acc*100:.2f}% | Test Acc={test_acc*100:.2f}% | Params={n_params:,}")
            return model, None, record
        except Exception as e:
            print(f"[Resume] Error loading existing model ({e}). Re-training...")

    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=TRAIN_CONFIG["early_stop_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=TRAIN_CONFIG["reduce_lr_factor"],
            patience=TRAIN_CONFIG["reduce_lr_patience"],
            min_lr=TRAIN_CONFIG["min_lr"],
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    print(f"\n{'='*60}")
    print(f"Training {model_class.upper()} | L={L} | HL={hl} | Train samples: {len(X_train)}")
    print(f"{'='*60}")

    t0 = time.time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cbs,
        class_weight=class_weight,
        verbose=1,
    )
    wall_time = time.time() - t0

    # Ensure best weights are saved in both .keras and .h5 formats for compatibility
    h5_path = save_dir / f"{model_class}_L{L}_hl{hl}.h5"
    try:
        model.save(str(h5_path))
    except Exception as e:
        print(f"Note: H5 save fallback: {e}")

    # Evaluate on validation set
    val_metrics = model.evaluate(val_ds, verbose=0)
    val_loss, val_acc, val_prec, val_rec = val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3]

    # Evaluate on test set
    test_metrics = model.evaluate(test_ds, verbose=0)
    test_loss, test_acc, test_prec, test_rec = test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3]

    # Confusion matrix on test set
    y_pred_prob = model.predict(test_ds, verbose=0)
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    # Save normalized confusion matrix to CSV (Task 5.1)
    confusion_dir = TRAIN_CONFIG["paths"]["confusion"]
    cm_df = pd.DataFrame(
        cm_norm,
        index=["true_normal", "true_speeding"],
        columns=["pred_normal", "pred_speeding"],
    )
    cm_csv_path = confusion_dir / f"{model_class}_L{L}_hl{hl}.csv"
    cm_df.to_csv(cm_csv_path)

    # Best epoch
    val_losses = hist.history.get("val_loss", [])
    epoch_best = int(np.argmin(val_losses) + 1) if val_losses else len(val_losses)

    # Model size and parameters (Task 5.2)
    n_params = model.count_params()
    size_file = model_path if model_path.exists() else h5_path
    size_kb = os.path.getsize(size_file) / 1024.0 if size_file.exists() else 0.0

    record = {
        "model": model_class,
        "L_seq": L,
        "hl": hl,
        "val_acc": float(val_acc),
        "val_loss": float(val_loss),
        "val_prec": float(val_prec),
        "val_rec": float(val_rec),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "test_prec": float(test_prec),
        "test_rec": float(test_rec),
        "tn_rate": float(cm_norm[0, 0]),
        "fp_rate": float(cm_norm[0, 1]),
        "fn_rate": float(cm_norm[1, 0]),
        "tp_rate": float(cm_norm[1, 1]),
        "params": int(n_params),
        "size_kb": float(size_kb),
        "epoch_best": epoch_best,
        "epochs_trained": len(val_losses),
        "wall_time_s": float(wall_time),
    }

    print(f"Results {model_class.upper()} L={L} HL={hl}: Val Acc={val_acc*100:.2f}% | Test Acc={test_acc*100:.2f}% | Params={n_params:,} | Size={size_kb:.1f}KB")

    return model, hist, record

def run_grid_sweep():
    """
    Phase 4: Hyperparameter sweep (Fig. 5 grid)
    Sweeps L in [5, 10, 15, 20, 25] and HL in [1, 2, 3] for CNN
    plus GRU baseline comparison.
    """
    results_dir = TRAIN_CONFIG["paths"]["results"]
    reports_dir = TRAIN_CONFIG["paths"]["reports"]

    seq_lengths = TRAIN_CONFIG["seq_lengths"]
    hidden_layers = TRAIN_CONFIG["hidden_layers"]

    sweep_records = []

    print("\n" + "=" * 70)
    print("STARTING HYPERPARAMETER SWEEP (CNN ResNet-1D & GRU)")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Hidden layer depths: {hidden_layers}")
    print("=" * 70)

    # 1. Sweep CNN models
    for L in seq_lengths:
        for hl in hidden_layers:
            model, hist, rec = train_one(L, hl, model_class="cnn")
            sweep_records.append(rec)

    # 2. Sweep GRU baselines (e.g. L=5, 15, 25 with HL=1, 2 for Fig. 5 comparison)
    for L in [5, 15, 25]:
        for hl in [1, 2]:
            model, hist, rec = train_one(L, hl, model_class="gru")
            sweep_records.append(rec)

    # Save sweep.csv (Task 4.1)
    sweep_df = pd.DataFrame(sweep_records)
    sweep_csv_path = results_dir / "sweep.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"\nSaved sweep results to: {sweep_csv_path}")

    # Plot Fig. 5-style chart (Task 4.2)
    plot_fig5(sweep_df, reports_dir / "fig5_sweep.png")

    # Plot Model size vs HL & L (Task 5.2)
    plot_size_chart(sweep_df, reports_dir / "size_plot.png")

    print("\n--- Sweep Summary Table ---")
    cols_to_show = ["model", "L_seq", "hl", "val_acc", "test_acc", "params", "size_kb", "epoch_best", "wall_time_s"]
    print(sweep_df[cols_to_show].to_string(index=False))

    return sweep_df

def plot_fig5(df, out_path):
    """
    Plot Fig. 5 style accuracy curves: Sequence Length vs Accuracy per HL.
    """
    plt.figure(figsize=(10, 6))
    cnn_df = df[df["model"] == "cnn"]
    gru_df = df[df["model"] == "gru"]

    hl_colors = {1: "tab:blue", 2: "tab:green", 3: "tab:red"}
    hl_markers = {1: "o-", 2: "s-", 3: "^-"}

    for hl in sorted(cnn_df["hl"].unique()):
        sub = cnn_df[cnn_df["hl"] == hl].sort_values("L_seq")
        plt.plot(
            sub["L_seq"],
            sub["test_acc"] * 100,
            hl_markers[hl],
            color=hl_colors[hl],
            lw=2,
            markersize=7,
            label=f"ResNet-1D (HL{hl})",
        )

    # Overlay GRU if present
    for hl in sorted(gru_df["hl"].unique()):
        sub = gru_df[gru_df["hl"] == hl].sort_values("L_seq")
        plt.plot(
            sub["L_seq"],
            sub["test_acc"] * 100,
            "--",
            color=hl_colors.get(hl, "tab:gray"),
            alpha=0.6,
            label=f"GRU (HL{hl})",
        )

    plt.title("SecureRide Detection Accuracy vs. Sequence Length (Fig. 5 Reproduction)", fontsize=13, fontweight="bold")
    plt.xlabel("Sequence Length ($L_{seq}$ in seconds at 1 Hz)", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.ylim(80, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Fig. 5 plot to: {out_path}")

def plot_size_chart(df, out_path):
    """
    Plot Model Size (KB) and Parameter counts vs. HL.
    """
    cnn_df = df[df["model"] == "cnn"].drop_duplicates(subset=["hl"]).sort_values("hl")
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("Number of Residual Blocks (HL)", fontsize=11)
    ax1.set_ylabel("Parameters", color=color, fontsize=11)
    ax1.bar(cnn_df["hl"] - 0.15, cnn_df["params"], width=0.3, color=color, label="Parameters")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Model Size (KB)", color=color, fontsize=11)
    ax2.bar(cnn_df["hl"] + 0.15, cnn_df["size_kb"], width=0.3, color=color, label="Size (KB)")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("SecureRide Model Complexity vs. Hidden Layers (Paper Fig. 6b)", fontsize=13, fontweight="bold")
    plt.xticks(cnn_df["hl"], [f"HL{h}" for h in cnn_df["hl"]])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved model size plot to: {out_path}")

if __name__ == "__main__":
    run_grid_sweep()
