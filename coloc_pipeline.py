from colocalizer_functions import *
from os.path import basename, splitext
import glob
import sys

def pyColocalizer(img, ch1, ch2, threshold, graph=True):
    img_name = splitext(basename(img))[0]
    images = loadImage(img)
    df, img_shape = contourImage(images, ch1, ch2)
    df = applyThreshold(df, threshold)
    M1, M2, tM1, tM2 = MandersCoef(df)
    predictions, rsquared, coef, pearson = colocRegression(df)

    df = residualReconstitutor(df, predictions)
    if graph:
        colocGrapher(df, threshold, img_shape, predictions, rsquared, img_name)
    print(img_name)
    return(img_name, rsquared, coef[0][0], pearson[0], tM1, tM2)

def folderColocalizer(folder, chan1, chan2, threshold, graph=True, foldername=None):
    """Function to facilitate full folder colocalization analysis."""
    tif_pattern = folder + "/*.tif"
    tiff_list = glob.glob(tif_pattern)
    output_list = []
    try:
        for img in tiff_list:
            output_list.append(pyColocalizer(img, chan1, chan2, threshold, graph))
    except ValueError:
        pass

    output_df = pd.DataFrame(output_list, columns = ["Name", "rsquared", "coef", "Pearson", "tM1", "tM2"])

    if not foldername:
        foldername = splitext(basename(folder))[0]

    output_df.to_csv("{}_{}.csv".format(threshold, foldername), index=False)

### USAGE EXAMPLE
# correlate channels 1 and 2 with 0 thresholding and no graph output
# folderColocalizer("Mouse1_GM130_TGN", 2, 4, 0, graph=False, foldername="Mouse1_GM130_TGN_1_2")

folder = sys.argv[1]
print(folder)
folderColocalizer(folder, 2, 4, 0, graph=False, foldername="TNFa_coloc")
