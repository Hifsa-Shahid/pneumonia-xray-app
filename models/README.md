# Model files

Put these four files in this folder before deploying:

| File | Where it comes from |
|---|---|
| `resnet50.pt` | `pneumonia_project/models/` on your Google Drive (shrunk with `scripts/shrink_checkpoints.py`) |
| `densenet121.pt` | same |
| `efficientnet_b0.pt` | same |
| `app_config.json` | `pneumonia_project/app_config.json` (temperatures, thresholds, metrics) |

Shrink and copy everything in one step:

```bash
python scripts/shrink_checkpoints.py --src "path/to/pneumonia_project/models" --dst models \
    --config "path/to/pneumonia_project/app_config.json"
```

If `app_config.json` is missing the app still runs, but without calibration (temperature 1.0) and without the metrics table.
