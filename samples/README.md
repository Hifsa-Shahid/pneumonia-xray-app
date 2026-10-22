# Sample X-rays for the in-app "Try a sample" tab

Put **two normal** and **two pneumonia** X-rays here:

```
samples/normal/      <- 2 images (PNG or JPG)
samples/pneumonia/   <- 2 images (PNG or JPG)
```

The app lists every image in these folders, mixes the two classes, hides the labels until a sample is analysed,
and then shows the dataset label next to the model's result.

## Picking the images

Use images from the dataset's **test** split, never from training. Random picks are the most honest choice. In your Colab
notebook (after the dataset cell, where `ROOT` is defined) run:

```python
import random, shutil
from pathlib import Path
random.seed(0)
out = Path("/content/samples"); 
for cls in ("NORMAL", "PNEUMONIA"):
    (out / cls.lower()).mkdir(parents=True, exist_ok=True)
    for f in random.sample(sorted((ROOT / "test" / cls).glob("*.jpeg")), 2):
        shutil.copy(f, out / cls.lower() / f.name)
!cd /content && zip -rq samples.zip samples
from google.colab import files; files.download("/content/samples.zip")
```

Unzip and copy the two folders into this `samples/` directory. If you hand-pick images instead, say so in the README.
