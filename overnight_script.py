from coloc_pipeline import *
import numpy as np

neg_ctrl = "C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Neg ctrl\\SP8_Confocal_dapi_CD9\\all_tifs"
pos_ctrl_002 = "C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Pos ctrl\\tetra\\downsampled\\002"
pos_ctrl_003 = "C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Pos ctrl\\tetra\\downsampled\\003"
vamp = "C:\\Users\\Peter\\Dropbox\\Colocalization methods\\data\\Single Cells"


for thres in np.arange(0, 1.1, 0.1):
    folderColocalizer(neg_ctrl, 1, 3, thres, graph=False)
    print('neg done')
    folderColocalizer(pos_ctrl_002, 1, 2, thres, graph=False)
    print('pos 002 done')
    folderColocalizer(pos_ctrl_003, 1, 2, thres, graph=False)
    print('pos 003 done')
    folderColocalizer(vamp, 1, 2, thres, graph=False)
    print('vamp done')
