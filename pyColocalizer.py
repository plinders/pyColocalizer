# pyColocalizer
# Rewrite of ColocalizeR in python

# imports
import tifffile as tiff
import numpy as np
import pandas as pd
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import glob
import os
from os.path import basename, splitext


def imgToDataframe(images):
    d = {}
    for i in range(len(images)):
        d["chan{}".format(i + 1)] = images[i].flatten()
    return pd.DataFrame(d)

def colocGrapher(df):
    plt.figure(figsize=(12,12))
    G = gridspec.GridSpec(4, 4)

    ax_1 = plt.subplot(G[0,0])
    ax_1.imshow(df["chan1"].values.reshape(img_shape), cmap = plt.cm.gray)
    axisRemover()
    ax_1.set(title = "{} original".format(df.columns[0]))

    ax_2 = plt.subplot(G[0,1])
    ax_2.imshow(df["chan2"].values.reshape(img_shape), cmap = plt.cm.gray)
    axisRemover()
    ax_2.set(title = "{} original".format(df.columns[1]))

    ax_3 = plt.subplot(G[0,2])
    sns.distplot(df["chan1"], axlabel=df.columns[0])
    ax_3.set(title = "Original")

    ax_4 = plt.subplot(G[0,3])
    sns.distplot(df["chan2"], axlabel=df.columns[1])
    ax_4.set(title = "Original")

    ax_5 = plt.subplot(G[1,0])
    ax_5.imshow(df_fix["chan1"].values.reshape(img_shape), cmap = plt.cm.gray)
    axisRemover()
    ax_5.set(title = "{} filtered".format(df.columns[0]))

    ax_6 = plt.subplot(G[1,1])
    ax_6.imshow(df_fix["chan2"].values.reshape(img_shape), cmap = plt.cm.gray)
    axisRemover()
    ax_6.set(title = "{} filtered".format(df.columns[1]))

    ax_7 = plt.subplot(G[1,2])
    sns.distplot(df_fix["chan1"].dropna(), axlabel=df.columns[0])
    ax_7.set(title = "Filtered")

    ax_8 = plt.subplot(G[1,3])
    sns.distplot(df_fix["chan2"].dropna(), axlabel=df.columns[1])
    ax_8.set(title = "Filtered")

    ax_9 = plt.subplot(G[2:, :2])
    plt.scatter(df_fix["chan1"].dropna(), df_fix["chan2"].dropna())
    plt.plot(df_X, predictions, color = "red")
    plt.text(x = 1, y = df_fix["chan2"].max(), s = "$R^2$: {}".format(round(rsquared, 3)), fontsize = 14)
    ax_9.set(xlabel = df.columns[0], ylabel = df.columns[1], title = "Correlation")

    ax_10 = plt.subplot(G[2:, 2:])
    sns.distplot((df_y - predictions), bins = 50)
    ax_10.set(title = "Residuals")

    plt.tight_layout()

    plt.savefig("{}.pdf".format(splitext(basename(input_tiff))[0]), dpi = 300, papertype = "a4")

def pyColocalizer(input_tiff, ch1, ch2, threshold):
    with tiff.TiffFile(input_tiff) as tif:
        images = tif.asarray()
        metadata = tif[0].tags

    ch1 = ch1 - 1
    ch2 = ch2 - 1

    img_shape = images[0].shape

    df = imgToDataframe(images)

    df_filter = df.copy()

    df_filter[(df_filter.iloc[:, ch1] < (threshold * df_filter.iloc[:, ch1].max())) &
              (df_filter.iloc[:, ch2] < (threshold * df_filter.iloc[:, ch2].max()))] = np.nan

    df_X = df_filter.iloc[:, ch1].dropna().values.reshape(-1, 1)
    df_y = df_filter.iloc[:, ch2].dropna().values.reshape(-1, 1)

    lm = LinearRegression()
    lm.fit(X = df_X, y = df_y)
    predictions = lm.predict(df_X)
    rsquared = r2_score(df_y, predictions)
    coef = lm.coef_

    img_name = splitext(basename(input_tiff))[0]

    colocGrapher()
    return(img_name, rsquared, coef[0][0])


pyColocalizer("data/test1.tif", 1, 2, 0.1)

folder = "data"
os.getcwd()

tif_pattern = folder + "/*.tif"
tiff_list = glob.glob(tif_pattern)

output_list = []
os.chdir("..")
for img in tiff_list:
    output_list.append(pyColocalizer(img, 1, 2, 0.1))

output_df = pd.DataFrame(output_list, columns = ["Name", "rsquared", "coef"])
