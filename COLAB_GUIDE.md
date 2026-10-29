# SecureRide on Google Colab

## Cell 1 — Upload and extract the updated ZIP
```python
from google.colab import files
uploaded = files.upload()  # choose SecureRide_Final_Updated_v2.zip
zip_name = next(name for name in uploaded if name.lower().endswith(".zip"))
```

```python
import os, zipfile, shutil
PROJECT_DIR = "/content/SecureRide_Final_Project"
if os.path.exists(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)
with zipfile.ZipFile(zip_name) as z:
    z.extractall("/content")
# The ZIP contains a top-level SecureRide_Final_Project/ directory.
os.chdir(PROJECT_DIR)
print("Project:", os.getcwd())
print("Files:", sorted(os.listdir("."))[:25])
```

## Cell 2 — Install dependencies
```python
%cd /content/SecureRide_Final_Project
!pip -q install -r requirements.txt
```
If Colab reports that packages were upgraded and asks for a runtime restart, restart and rerun Cell 1 (the uploaded ZIP may need to be uploaded again), then continue.

## Cell 3 — Verify runtime and files
```python
import torch, os
print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
assert os.path.exists("data"), "Missing data/ directory"
assert os.path.exists("train.py")
```
For GPU: Runtime → Change runtime type → T4 GPU (if available).

## Cell 4 — Run unit tests
```python
!python -m unittest test_secureride.py test_contributions.py
```

## Cell 5 — Train both original tasks (CNN)
```python
!python train.py --backend torch --arch cnn \
  --tasks speeding sidewalk \
  --lengths 5 10 20 --hls 1 2 \
  --max-train 4000 --max-eval 3000 --epochs 15 --seed 0
```

## Cell 6 — View the resulting accuracy table
```python
import pandas as pd
metrics = pd.read_csv("results/metrics_torch.csv")
display(metrics[["task","arch","L","hl","val_acc","test_acc","test_prec","test_rec","test_f1","exec_ms"]]
        .sort_values(["task","test_acc"], ascending=[True,False]))
```
`test_acc` is a final held-out test metric. Select configurations using validation metrics, not test accuracy.

## Cell 7 — Run class-imbalance experiments
```python
!python imbalance.py --backend torch --task sidewalk --L 10 --hl 2 \
  --prevalences 0.5 0.05 0.01 --weights none balanced --seeds 3
!python imbalance.py --backend torch --task speeding --L 10 --hl 2 \
  --prevalences 0.5 0.05 0.01 --weights none balanced --seeds 3
```
For a quicker first check, use `--backend sklearn` or reduce `--seeds` to 1. Inspect the exact output filenames printed by the scripts.

## Cell 8 — Run temporal-smoothing experiments
```python
!python smoothing.py --backend torch --task sidewalk --ns 1 3 5 7
!python smoothing.py --backend torch --task speeding --ns 1 3 5 7
```
This uses saved quick/accurate checkpoints. Ensure the expected model files exist in `models/`. Compare false alarms and detection delay along with precision/recall/F1.

## Cell 9 — Package the updated project, including new results and docs
```python
import os, zipfile
from google.colab import files

zip_path = "/content/SecureRide_Final_Updated_Colab.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for folder, dirs, filenames in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv"}]
        for filename in filenames:
            full = os.path.join(folder, filename)
            rel = os.path.relpath(full, "/content")
            z.write(full, rel)
print("ZIP size (MB):", round(os.path.getsize(zip_path)/(1024**2), 2))
files.download(zip_path)
```

### Notes
- The provided data is synthetic. Avoid describing the sample traces as real-world recordings.
- Preserve original results before rerunning a sweep if you want a run-to-run comparison; scripts may overwrite result CSVs.
- Report the exact command, seed, configuration, backend, and output CSV for each reported metric.
