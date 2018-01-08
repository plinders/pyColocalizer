from colocalizer_functions import *
from os.path import basename, splitext
import glob

def pyColocalizer(img, ch1, ch2, threshold):
    img_name = splitext(basename(img))[0]
    images = loadImage(img)
    df, img_shape = contourImage(images, chan1, chan2)
    df = applyThreshold(df, threshold)
    MA, MB = MandersCoef(df, img_shape)
    predictions, rsquared, coef, pearson = colocRegression(df)

    df = residualReconstitutor(df, predictions)
    colocGrapher(df, threshold, img_shape, predictions, rsquared, img_name)
    return(img_name, rsquared, coef[0][0], pearson[0], MA, MB)

def folderColocalizer(folder, chan1, chan2, threshold):
    """Function to facilitate full folder colocalization analysis."""
    tif_pattern = folder + "/*.tif"
    tiff_list = glob.glob(tif_pattern)
    output_list = []
    for img in tiff_list:
        output_list.append(pyColocalizer(img, chan1, chan2, threshold))

    output_df = pd.DataFrame(output_list, columns = ["Name", "rsquared", "coef", "Pearson", "M1", "M2"])

    foldername = splitext(basename(folder))[0]

    output_df.to_csv("{}_{}.csv".format(threshold, foldername), index=False)
