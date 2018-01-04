import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

    return(pd.DataFrame(d))


def axisRemover():
    """Short function to remove axis ticks from plt plots."""
    plt.xticks(())
    plt.yticks(())


def residualReconstitutor(final_df, predictions):
    """Function to reconsitute an image from residuals"""
    residuals = abs(final_df["contour_chan2"].dropna() - predictions)
    final_df['residuals'] = pd.DataFrame(residuals, index = final_df["contour_chan1"].dropna().index)
    return(final_df)


def colocGrapher(final_df, img_shape, predictions, rsquared, img_name):
    """Function to graphs data from pyColocalizer."""
    #set limits for residual colorbar
    minval = abs(final_df['residuals']).min()
    maxval = abs(final_df['residuals']).max()

    plt.figure(figsize = (12, 12))
    G = gridspec.GridSpec(4, 4)

    #chan1 raw
    ax1 = plt.subplot(G[0, 0])
    ax1.imshow(final_df["orig_chan1"].values.reshape(img_shape), cmap = "gray")
    axisRemover()
    ax1.set(title="Channel 1 original")

    #chan2 raw
    ax2 = plt.subplot(G[0, 1])
    ax2.imshow(final_df["orig_chan2"].values.reshape(img_shape), cmap = "gray")
    axisRemover()
    ax2.set(title="Channel 2 original")

    #chan1 distribution
    ax3 = plt.subplot(G[0, 2])
    sns.distplot(final_df["orig_chan1"], axlabel="Channel 1")
    ax3.set(title="Original")

    #chan2 distribution
    ax4 = plt.subplot(G[0, 3])
    sns.distplot(final_df["orig_chan2"], axlabel="Channel 2")
    ax4.set(title="Original")

    #chan1 filtered
    ax5 = plt.subplot(G[1, 0])
    ax5.imshow(final_df["contour_chan1"].values.reshape(img_shape), cmap="gray")
    axisRemover()
    ax5.set(title="Channel 1 filtered")

    #chan2 filtered
    ax6 = plt.subplot(G[1, 1])
    ax6.imshow(final_df["contour_chan2"].values.reshape(img_shape), cmap="gray")
    axisRemover()
    ax6.set(title="Channel 2 filtered")

    #chan1 filtered distribution
    ax7 = plt.subplot(G[1, 2])
    sns.distplot(final_df["contour_chan1"].dropna(), axlabel = "Channel 1")
    ax7.set(title="Filtered")

    #chan2 filtered distribution
    ax8 = plt.subplot(G[1, 3])
    sns.distplot(final_df["contour_chan2"].dropna(), axlabel = "Channel 2")
    ax8.set(title="Filtered")

    #data scatter plot + regression line
    ax9 = plt.subplot(G[2:, :2])
    plt.scatter(final_df["contour_chan1"].dropna(), final_df["contour_chan2"].dropna())
    plt.plot(final_df["contour_chan1"].dropna(), predictions, color="red")
    plt.text(x=1, y = final_df["contour_chan2"].max(), s="$R^2$: {}".format(
        round(rsquared, 3)), fontsize=14)
    ax9.set(xlabel="Channel 1", ylabel = "Channel 2", title = "Correlation")

    #fig with residuals/colocalization areas
    ax10 = plt.subplot(G[2:, 2:])
    cax = ax10.imshow(abs(final_df['residuals']).values.reshape(img_shape), cmap = "Blues")
    cbar = fig.colorbar(cax, ticks = [minval, maxval])
    cbar.ax.set_yticklabels(['Max colocalization', 'Min colocalization'])
    axisRemover()

    plt.tight_layout()

    plt.savefig("{}.pdf".format(img_name), dpi=300, papertype="a4")

    plt.close()
