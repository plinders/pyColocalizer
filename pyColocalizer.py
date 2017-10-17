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


def imgToDataframe(images):
    d = {}
    for i in range(len(images)):
        d["chan{}".format(i + 1)] = images[i].flatten()
    return pd.DataFrame(d)


def pyColocalizer(input_tiff, ch1, ch2, threshold):
    with tiff.TiffFile(input_tiff) as tif:
        images = tif.asarray()
        metadata = tif[0].tags

    img_shape = images[0].shape

    df = imgToDataframe(images)

    df_filter = df.copy()

    df_filter[(df_filter[ch1] < (threshold * df_filter[ch1].max())) &
              (df_filter[ch2] < (threshold * df_filter[ch2].max()))] = np.nan

    df_X = df_filter[ch1].dropna().values.reshape(-1, 1)
    df_y = df_filter[ch2].dropna().values.reshape(-1, 1)

    lm = LinearRegression()
    lm.fit(X = df_X, y = df_y)
    predictions = lm.predict(df_X)
    rsquared = r2_score(df_y, predictions)
    coef = lm.coef_

    return(rsquared, coef)
