import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image

def load_image(img_file):
    with tiff.TiffFile(img_file) as tif:
        images = tif.asarray().astype(float)
    return(images)

def contour_image(raw_images, ch1, ch2):
    (chan1, chan2) = ch1 - 1, ch2 - 1

    raw_img = raw_images[[chan1, chan2]]
    contour_img = raw_img.copy()

    thresh1 = threshold_otsu(contour_img[0])
    binary1 = contour_img[0] > thresh1
    chull1 = convex_hull_image(binary1)

    thresh2 = threshold_otsu(contour_img[1])
    binary2 = contour_img[0] > thresh2
    chull2 = convex_hull_image(binary2)

    if sum(chull1.flatten()) > sum(chull2.flatten()):
        contour_img[0][chull1 == False] = np.nan
        contour_img[1][chull1 == False] = np.nan

    else:
        contour_img[0][chull2 == False] = np.nan
        contour_img[1][chull2 == False] = np.nan

    d = {}
    for i, data in enumerate(raw_img):
        d["orig_chan{}".format(i + 1)] = raw_img[i].flatten()
        d["contour_chan{}".format(i + 1)] = contour_img[i].flatten()

    return pd.DataFrame(d)
