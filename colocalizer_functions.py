"""colocalizer_functions.py
These serve as helper functions used in coloc_pipeline.py
"""
import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def loadImage(img_file):
    """Loads open tiff connection to n-dimensional numpy arrays
    """
    with tiff.TiffFile(img_file) as tif:
        images = tif.asarray().astype(float)
    return(images)


def contourImage(raw_images, ch1, ch2):
    """Detects largest object in image and draws a contour around it.
    All information in both channels outside the contour are set to NaN.
    In addition, a pandas DataFrame is generated.
    """
    #Transform channel numbers to indices
    (chan1, chan2) = ch1 - 1, ch2 - 1
    #Save image dimensions to be able to reconsitute images from 1d-arrays
    img_shape = raw_images[0].shape

    raw_img = raw_images[[chan1, chan2]]
    contour_img = raw_img.copy()
    #Binary masking of both channels, make convex hull for contouring
    try:
        thresh1 = threshold_otsu(contour_img[0])
        binary1 = contour_img[0] > thresh1
        chull1 = convex_hull_image(binary1)
        thresh2 = threshold_otsu(contour_img[1])
        binary2 = contour_img[1] > thresh2
        chull2 = convex_hull_image(binary2)

        if sum(chull1.flatten()) > sum(chull2.flatten()):
            #Any pixel where no data of the convex hull is, is set to NaN
            contour_img[0][chull1 == False] = np.nan
            contour_img[1][chull1 == False] = np.nan

        else:
            contour_img[0][chull2 == False] = np.nan
            contour_img[1][chull2 == False] = np.nan

    except ValueError:
        pass

    #Populate dictionary with data to return the final DataFrame
    d = {}
    for i, data in enumerate(raw_img):
        d["orig_chan{}".format(i + 1)] = raw_img[i].flatten()
        d["contour_chan{}".format(i + 1)] = contour_img[i].flatten()

    return(pd.DataFrame(d), img_shape)


def applyThreshold(final_df, threshold):
    """Applies a user-defined threshold value to the previously imported data.
    Threshold is received as a fraction of 1.
    In addition, sets areas that are 0 in both channels to NaN.
    """
    #Make copy of channels of interest, just in case
    thres_df = final_df.loc[:,['contour_chan1', 'contour_chan2']].copy()

    thres_df = thres_df - thres_df.min()

    thres_df[(thres_df.loc[:, 'contour_chan1'] == 0) & (thres_df.loc[:, 'contour_chan2'] == 0)] = np.nan

    thres_df[(thres_df.loc[:, 'contour_chan1'] < (threshold * thres_df.loc[:, 'contour_chan1'].max())) &
             (thres_df.loc[:, 'contour_chan2'] < (threshold * thres_df.loc[:, 'contour_chan2'].max()))] = np.nan
    #Replace channels of interest in original dataframe by the thresholded ones
    final_df[['thres_chan1', 'thres_chan2']] = thres_df

    return(final_df)


def MandersCoef(final_df):
    """Computes Manders' coefficients.
    """

    M1 = np.nansum(final_df[final_df['contour_chan2'] > 0]['contour_chan1'].values) / np.nansum(final_df['contour_chan1'])
    M2 = np.nansum(final_df[final_df['contour_chan1'] > 0]['contour_chan2'].values) / np.nansum(final_df['contour_chan2'])

    tM1 = np.nansum(final_df[final_df['thres_chan2'] > 0]['thres_chan1'].values) / np.nansum(final_df['thres_chan1'])
    tM2 = np.nansum(final_df[final_df['thres_chan1'] > 0]['thres_chan2'].values) / np.nansum(final_df['thres_chan2'])

    return(M1, M2, tM1, tM2)


def colocRegression(final_df):
    """Performs linear regression on the processed images.
    Returns predicted values of channel 2, R-squared, coefficient and Pearson's R.
    """
    #Make 1d-arrays for the linear model.
    df_X = final_df['thres_chan1'].dropna().values.reshape(-1, 1)
    df_y = final_df['thres_chan2'].dropna().values.reshape(-1, 1)

    lm = LinearRegression()
    lm.fit(X = df_X, y = df_y)
    predictions = lm.predict(df_X)
    rsquared = r2_score(df_y, predictions)
    coef = lm.coef_
    pearson = pearsonr(df_X, df_y)

    return(predictions, rsquared, coef, pearson)


def residualReconstitutor(final_df, predictions):
    """Reconstitues an image from residuals.
    """
    residuals = abs(final_df["thres_chan2"].dropna().values.reshape(-1, 1) - predictions)
    final_df['residuals'] = pd.DataFrame(residuals, index = final_df["thres_chan1"].dropna().index)
    return(final_df)


def axisRemover():
    """Removes axis ticks from plt plots."""
    plt.xticks(())
    plt.yticks(())


def colocGrapher(final_df, threshold, img_shape, predictions, rsquared, img_name):
    """Graphs data from pyColocalizer."""
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
    ax5.imshow(final_df["thres_chan1"].values.reshape(img_shape), cmap="gray")
    axisRemover()
    ax5.set(title="Channel 1 filtered")

    #chan2 filtered
    ax6 = plt.subplot(G[1, 1])
    ax6.imshow(final_df["thres_chan2"].values.reshape(img_shape), cmap="gray")
    axisRemover()
    ax6.set(title="Channel 2 filtered")

    #chan1 filtered distribution
    ax7 = plt.subplot(G[1, 2])
    sns.distplot(final_df["thres_chan1"].dropna(), axlabel = "Channel 1")
    ax7.set(title="Filtered")

    #chan2 filtered distribution
    ax8 = plt.subplot(G[1, 3])
    sns.distplot(final_df["thres_chan2"].dropna(), axlabel = "Channel 2")
    ax8.set(title="Filtered")

    #data scatter plot + regression line
    ax9 = plt.subplot(G[2:, :2])
    plt.hexbin(final_df["thres_chan1"].dropna(), final_df["thres_chan2"].dropna(), bins='log', gridsize=25, cmap='Blues')
    plt.plot(final_df["thres_chan1"].dropna(), predictions, color="red")
    #plt.text(x=1, y = final_df["contour_chan2"].max(), s="$R^2$: {}".format(
    #    round(rsquared, 3)), fontsize=14)
    ax9.set(xlabel="Channel 1", ylabel = "Channel 2", title = "Correlation $R^2$: {}".format(
        round(rsquared, 3)))

    #fig with residuals/colocalization areas
    ax10 = plt.subplot(G[2:, 2:])
    cax = ax10.imshow(abs(final_df['residuals']).values.reshape(img_shape), cmap = "Blues")
    cbar = plt.colorbar(cax, ticks = [minval, maxval])
    cbar.ax.set_yticklabels(['Min colocalization', 'Max colocalization'])
    ax10.set(title = "Colocalization")
    axisRemover()

    plt.tight_layout()

    plt.savefig("{}_{}.pdf".format(threshold, img_name), dpi=300, papertype="a4")

    plt.close()
