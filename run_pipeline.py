import subprocess
import sys
import time
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    scripts = [
        "01_inspect.py",
        "02_clean.py",
        "03_features.py",
        "04_label.py",
        "05_window.py",
        "06_split.py",
        "07_augment.py",
        "08_package.py",
    ]

    print("=" * 70)
    print("SecureRide End-to-End Data Pipeline Execution")
    print("=" * 70)

    total_start = time.time()

    for s in scripts:
        script_path = root / "src" / s
        print(f"\n>>> Running {s}...")
        t0 = time.time()
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(root))
        if res.returncode != 0:
            print(f"Error running {s}! Aborting pipeline.")
            sys.exit(res.returncode)
        dt = time.time() - t0
        print(f">>> Finished {s} in {dt:.2f}s")

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"Pipeline completed successfully in {total_time:.2f} seconds.")
    print("Artifacts saved to data/final/ and reports/final_qa.md")
    print("=" * 70)

if __name__ == "__main__":
    main()
