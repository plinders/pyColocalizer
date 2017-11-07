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

    df_filter[df_filter == 0] = np.nan

    df_fix = df_filter - df_filter.min()

    df_fix[(df_fix.iloc[:, ch1] < (threshold * df_fix.iloc[:, ch1].max())) &
           (df_fix.iloc[:, ch2] < (threshold * df_fix.iloc[:, ch2].max()))] = np.nan

    df_mean = df_fix / np.mean(df_fix)

    df_X = df_mean.iloc[:, ch1].dropna().values.reshape(-1, 1)
    df_y = df_mean.iloc[:, ch2].dropna().values.reshape(-1, 1)

    lm = LinearRegression()
    lm.fit(X=df_X, y=df_y)
    predictions = lm.predict(df_X)
    rsquared = r2_score(df_y, predictions)
    coef = lm.coef_

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

        plt.savefig("{}.pdf".format(img_name), dpi=300, papertype="a4")

        plt.close()

    colocGrapher()
    return(img_name, rsquared, coef[0][0])


def folderColocalizer(folder, chan1, chan2, threshold):
    """Function to facilitate full folder colocalization analysis."""
    tif_pattern = folder + "/*.tif"
    tiff_list = glob.glob(tif_pattern)
    output_list = []
    for img in tiff_list:
        output_list.append(pyColocalizer(img, chan1, chan2, threshold))

    output_df = pd.DataFrame(output_list, columns=["Name", "rsquared", "coef"])
    output_df.to_csv(splitext(basename(folder))[0] + ".csv", index=False)


import os
os.chdir("C:\\Users\\Peter\\Documents\\Papers\\Colocalization methods\\data")

folderColocalizer("C:\\Users\\Peter\\Documents\\Papers\\Colocalization methods\\data\\Single Cells", 1, 2, 0.1)
