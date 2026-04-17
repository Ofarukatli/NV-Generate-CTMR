import os
import re

scripts_dir = r"C:\Users\isd.omer.atli\Desktop\bilkent\server\NV-Generate-CTMR\scripts"

patterns = [
    r"\w+\.shape\[[2-4]\]", # shape[2], shape[3], shape[4]
    r"spacing\[[0-3]\]",
    r"dim\[[0-3]\]",
    r"output_size\[[0-3]\]",
    r"3d",
]

for root, _, files in os.walk(scripts_dir):
    for f in files:
        if f.endswith(".py"):
            filepath = os.path.join(root, f)
            with open(filepath, "r", encoding="utf-8") as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    for p in patterns:
                        if re.search(p, line, re.IGNORECASE):
                            print(f"{f}:{i+1} : {line.strip()}")
                            break
