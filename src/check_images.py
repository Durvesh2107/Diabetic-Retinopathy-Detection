import os
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

df = pd.read_csv("data/train.csv")
df["id_code"] = df["id_code"].astype(str) + ".png"

image_dir = "data/train_images"
bad_images = []

for i, image_name in enumerate(df["id_code"]):
    image_path = os.path.join(image_dir, image_name)

    try:
        with Image.open(image_path) as img:
            img.load()
    except Exception as e:
        print("Bad image:", image_name, "| Error:", e)
        bad_images.append(image_name)

    if i % 100 == 0:
        print("Checked:", i)

print("Total bad images:", len(bad_images))
print(bad_images)

pd.DataFrame({"bad_images": bad_images}).to_csv("data/bad_images.csv", index=False)