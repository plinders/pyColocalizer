"""Script to perform colocalization analysis on two channels of a multi-channel tif image."""

# imports
from __future__ import division
from os.path import basename, splitext
import glob
import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def pyColocalizer(input_tiff, ch1, ch2, threshold):
    """Function to calculate linear regression between two channels of interest."""
    with tiff.TiffFile(input_tiff) as tif:
        images = tif.asarray()

    ch1 = ch1 - 1
    ch2 = ch2 - 1

    img_shape = images[0].shape

    def imgToDataframe():
        """Function to transform all channels of open tif connection to columns in a DataFrame."""
        d = {}
        for i, data in enumerate(images):
            d["chan{}".format(i + 1)] = images[i].flatten()
        return pd.DataFrame(d)

    df = imgToDataframe()

    df_filter = df.copy()

    df_filter[(df_filter.iloc[:, ch1] == 0) & (df_filter.iloc[:, ch2] == 0)] = np.nan

    df_fix = df_filter - df_filter.min()

    df_fix[(df_fix.iloc[:, ch1] < (threshold * df_fix.iloc[:, ch1].max())) &
           (df_fix.iloc[:, ch2] < (threshold * df_fix.iloc[:, ch2].max()))] = np.nan

    # if (df_fix.iloc[:, ch1].max() >= 65535) | (df_fix.iloc[:, ch2].max() >= 65535):
    #     df_fix[(df_fix.iloc[:, ch1] >= 65535) | (df_fix.iloc[:, ch2] >= 65535) ] = np.nan
    # elif (df_fix.iloc[:, ch1].max() >= 4095) | (df_fix.iloc[:, ch2].max() >= 4095):
    #     df_fix[(df_fix.iloc[:, ch1] >= 4095) | (df_fix.iloc[:, ch2] >= 4095) ] = np.nan
    # elif (df_fix.iloc[:, ch1].max() >= 255) | (df_fix.iloc[:, ch2].max() >= 255):
    #     df_fix[(df_fix.iloc[:, ch1] >= 255) | (df_fix.iloc[:, ch2] >= 255) ] = np.nan


    #normalize by subtracting mean
    df_mean = df_fix - np.mean(df_fix)

    df_X = df_mean.iloc[:, ch1].dropna().values.reshape(-1, 1)
    df_y = df_mean.iloc[:, ch2].dropna().values.reshape(-1, 1)

    lm = LinearRegression()
    lm.fit(X=df_X, y=df_y)
    predictions = lm.predict(df_X)
    rsquared = r2_score(df_y, predictions)
    coef = lm.coef_

    residuals = df_y - predictions

    pearson = pearsonr(df_X, df_y)

    img_name = splitext(basename(input_tiff))[0]

    def axisRemover():
        """Short function to remove axis ticks from plt plots."""
        plt.xticks(())
        plt.yticks(())


    def colocGrapher():
        """Function to graphs data from pyColocalizer."""
        plt.figure(figsize=(12, 12))
        G = gridspec.GridSpec(4, 4)

        ax_1 = plt.subplot(G[0, 0])
        ax_1.imshow(df["chan1"].values.reshape(img_shape), cmap="gray")
        axisRemover()
        ax_1.set(title="{} original".format(df.columns[0]))

        ax_2 = plt.subplot(G[0, 1])
        ax_2.imshow(df["chan2"].values.reshape(img_shape), cmap="gray")
        axisRemover()
        ax_2.set(title="{} original".format(df.columns[1]))

        ax_3 = plt.subplot(G[0, 2])
        sns.distplot(df["chan1"], axlabel=df.columns[0])
        ax_3.set(title="Original")

        ax_4 = plt.subplot(G[0, 3])
        sns.distplot(df["chan2"], axlabel=df.columns[1])
        ax_4.set(title="Original")

        ax_5 = plt.subplot(G[1, 0])
        ax_5.imshow(df_mean["chan1"].values.reshape(img_shape), cmap="gray")
        axisRemover()
        ax_5.set(title="{} filtered".format(df.columns[0]))

        ax_6 = plt.subplot(G[1, 1])
        ax_6.imshow(df_mean["chan2"].values.reshape(img_shape), cmap="gray")
        axisRemover()
        ax_6.set(title="{} filtered".format(df.columns[1]))

        ax_7 = plt.subplot(G[1, 2])
        sns.distplot(df_mean["chan1"].dropna(), axlabel=df.columns[0])
        ax_7.set(title="Filtered")

        ax_8 = plt.subplot(G[1, 3])
        sns.distplot(df_mean["chan2"].dropna(), axlabel=df.columns[1])
        ax_8.set(title="Filtered")

        ax_9 = plt.subplot(G[2:, :2])
        plt.scatter(df_mean["chan1"].dropna(), df_mean["chan2"].dropna())
        plt.plot(df_X, predictions, color="red")
        plt.text(x=1, y=df_mean["chan2"].max(), s="$R^2$: {}".format(
            round(rsquared, 3)), fontsize=14)
        ax_9.set(xlabel=df.columns[0], ylabel=df.columns[1], title="Correlation")

        ax_10 = plt.subplot(G[2:, 2:])
        sns.distplot((df_y - predictions), bins=50)
        ax_10.set(title="Residuals")

        plt.tight_layout()

        plt.savefig("{}_{}.pdf".format(threshold, img_name), dpi=300, papertype="a4")

    plt.close()


    colocGrapher()
    return(img_name, rsquared, coef[0][0], pearson)


def folderColocalizer(folder, chan1, chan2, threshold):
    """Function to facilitate full folder colocalization analysis."""
    tif_pattern = folder + "/*.tif"
    tiff_list = glob.glob(tif_pattern)
    output_list = []
    for img in tiff_list:
        output_list.append(pyColocalizer(img, chan1, chan2, threshold))

    output_df = pd.DataFrame(output_list, columns=["Name", "rsquared", "coef", "pearson"])

    foldername = splitext(basename(folder))[0]

    output_df.to_csv("{}_{}.csv".format(threshold, foldername), index=False)

    #output_df.to_csv(threshold + "_" + splitext(basename(folder))[0] + ".csv", index=False)


import os
os.chdir("C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Reciprocal")
folderColocalizer("C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Neg ctrl\\tetra\\downsampled", 1, 2, 0)

os.listdir("C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Neg ctrl\\tetra\\002")

tetra_list = ["C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Neg ctrl\\tetra\\downsampled\\002", "C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Neg ctrl\\tetra\\downsampled\\003"]
for i, folder in enumerate(tetra_list):
    os.chdir(folder)
    folderColocalizer(folder, 1, 2, 0)

folderColocalizer("C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Reciprocal", 1, 2, 0.1)
folderColocalizer("C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Reciprocal", 2, 1, 0.1)




os.listdir("C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Neg ctrl\\SP8_Confocal_dapi_CD9\\Confocal_dapi_CD9")

for i in os.listdir("C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Neg ctrl\\SP8_Confocal_dapi_CD9\\Confocal_dapi_CD9"):
    folderColocalizer(i, 1, 3, 0)

folderColocalizer("C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Neg ctrl\\SP8_Confocal_dapi_CD9\\Confocal_dapi_CD9\\160208_experiment 160120.4_STX3_CD9", 1, 3, 0)

os.getcwd()

for i in np.arange(0, 1.1, 0.1):
    folderColocalizer("C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Pos ctrl", 1, 2, i)


folderColocalizer("C:\\Users\\Peter\\Dropbox\\ellen", 1, 2, 0)

pyColocalizer("171128_PL_J.1 TetraSpeck.lif - Tetraspeck_560_660_002-11.tif", 1, 2, 0.1)

np.arange(0, 1.1, 0.1)

test_list = glob.glob("*.tif")

for item in test_list:
    print item

from timeit import default_timer as timer

for item in test_list:
    outlist = []
    for i in range(100):
        start = timer()
        pyColocalizer(item, 1, 2, 0)
        end = timer()
        outlist.append((item, i, end - start))
    outdf = pd.DataFrame(outlist, columns=["Name", "n", "time"])
    outdf.to_csv("{}.csv".format(item), index=False)
