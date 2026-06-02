import os
import subprocess

folder_url = "https://drive.google.com/drive/u/1/folders/1k6SQ88HXatRffBYvx2L6yvtfxppBFKMS"
output_dir = "./videos"

def get_all_videos(folder, output):
    os.makedirs(output, exist_ok=True)

    subprocess.run([
        "gdown",
        "--folder",
        folder,
        "-O", output,
        "--no-cookies"
    ])

get_all_videos(folder_url, output_dir)