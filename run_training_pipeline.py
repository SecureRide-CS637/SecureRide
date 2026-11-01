import subprocess
import sys
import time
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    venv_python = root / ".venv" / "Scripts" / "python.exe"

    if not venv_python.exists():
        venv_python = Path(sys.executable)

    scripts = [
        "src/train.py",
        "src/recompute_metrics.py",
        "src/evaluate_qa.py",
        "src/select_and_export.py",
    ]

    print("=" * 70)
    print("SecureRide End-to-End Model Training Pipeline Execution")
    print(f"Python interpreter: {venv_python}")
    print("=" * 70)

    total_start = time.time()

    for s in scripts:
        script_path = root / s
        print(f"\n>>> Running {s}...")
        t0 = time.time()
        res = subprocess.run([str(venv_python), str(script_path)], cwd=str(root))
        if res.returncode != 0:
            print(f"Error running {s}! Aborting training pipeline.")
            sys.exit(res.returncode)
        dt = time.time() - t0
        print(f">>> Finished {s} in {dt:.2f}s")

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"Model Training Pipeline completed successfully in {total_time:.2f} seconds.")
    print("Deliverables saved to:")
    print("  - models/theta_quick.{h5,tflite}")
    print("  - models/theta_accurate.{h5,tflite}")
    print("  - models/manifest.json")
    print("  - results/sweep.csv")
    print("  - results/confusion/*.csv")
    print("  - reports/fig5_sweep.png, reports/size_plot.png, reports/training_qa.md, reports/model_selection.md")
    print("=" * 70)

if __name__ == "__main__":
    main()
