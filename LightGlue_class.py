# streamlit_app.py
import os
import streamlit as st
import glob
import torch
from pathlib import Path
from lightglue import viz2d
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

st.title("Image Matcher using LightGlue")

# Setup
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

parentfolder = "../all_cropped_icons"
all_img_files = glob.glob(parentfolder+"/*")

# Load all images and compute features
feat_per_img = []
feat_per_img_dict = dict()
img2ix = dict()

# pickle the features
print("Computing features for all images...")
featurefile = 'feat_per_img.pickle'
if os.path.exists(featurefile):
    with open(featurefile, 'rb') as handle:
        feat_per_img = pickle.load(handle)
    print("Loaded features from pickle file")
else:
    for idx, imgpath in tqdm(enumerate(all_img_files)):
        img = load_image(imgpath)
        feats = extractor.extract(img.to(device))
        feat_per_img.append([feats, imgpath, img])
        feat_per_img_dict[Path(imgpath).stem] = feats
        img2ix[Path(imgpath).name] = idx

    with open(featurefile, 'wb') as handle:
        pickle.dump(feat_per_img, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved features to pickle file")

# Function to find top-k matches for the query image
def find_top_k_matches(queryimg, k=5):
    queryfeats = extractor.extract(queryimg.to(device))
    all_matches = []

    for parseix, (feats, imgpath, img) in enumerate(feat_per_img):
        matches01 = matcher({"image0": queryfeats, "image1": feats})
        feats0, feats, matches01 = [
            rbd(x) for x in [queryfeats, feats, matches01]
        ]  # remove batch dimension

        num_matches = list(matches01["matches"].shape)[0]
        all_matches.append((num_matches, imgpath, img, feats0, feats, matches01))

    # Sort by number of matches and return top-k
    all_matches.sort(key=lambda x: x[0], reverse=True)
    return all_matches[:k]


# Display all image names for selection
image_names = [Path(imgpath).name for imgpath in all_img_files]
# selected_image_name = st.slider("Select an image:", image_names)
selected_image_index = st.slider("Select an image:", 0, len(image_names) - 1)
selected_image_name = image_names[selected_image_index]

output_dir = Path("output_images/")
# create output dir and ignore if exists
output_dir.mkdir(parents=True, exist_ok=True)

# Load the selected image
queryimg = load_image(os.path.join(parentfolder, selected_image_name))
st.image(Image.open(os.path.join(parentfolder, selected_image_name)).convert("RGB"),
         caption=f'Selected Image: {selected_image_name}')

# Find top-k matches and display them
top_k_matches = find_top_k_matches(queryimg, k=5)
for ix, (num_matches, matched_imgpath, matched_img, feats0, feats, matches01) in enumerate(top_k_matches):
    kpts0, kpts1, matches = feats0["keypoints"], feats["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # Plotting and displaying
    axes = viz2d.plot_images([queryimg, matched_img])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    # Save the figure
    output_path = os.path.join("output_images/", f"viz2d_{Path(matched_imgpath).stem}_{ix}.png")
    plt.savefig(output_path)
    plt.close()

    # Display the saved image in Streamlit or provide a link to download
    st.image(output_path, caption=f'Matched Image with {num_matches} Keypoints from {matched_imgpath}', use_column_width=True)
    st.write(f"[Download the visualization]({output_path})")
