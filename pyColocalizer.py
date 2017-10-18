# pyColocalizer
# Rewrite of ColocalizeR in python

# imports
import tifffile as tiff
import numpy as np
import pandas as pd
from __future__ import division
import matplotlib.pyplot as pyplot
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import glob
import os


def imgToDataframe(images):
    d = {}
    for i in range(len(images)):
        d["chan{}".format(i + 1)] = images[i].flatten()
    return pd.DataFrame(d)

input_tiff = "data/test1.tif"
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

    img_name = input_tiff.split(".tif")[0]
    img_name = img_name.split("/")[-1]
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
