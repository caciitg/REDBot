import os
import gdown


os.makedirs("new_checkpoints", exist_ok=True)
url = "https://drive.google.com/uc?id=1GMj0u-EmawLpqKQ8NQJh_PC478aimG9D"
destination = "checkpoints/model.bin"
gdown.download(url, output=destination)
