import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
# from scipy.io import loadmat
# m=loadmat('1_file.mat')
# print(m)
# datas=h5py.File('1_file.mat','w')
datas=h5py.File('1_file.mat','r')
#print(datas['temp'])
data=datas['temp']
# print(data.shape)

# trainsample=datas['trainsample']
np.save('transample.npy',data)
dataa=np.load('transample.npy')
print(dataa.shape)
# plt.imshow(dataa[128])
# plt.pause(0.01)
# def mat_to_npy(matpath,imgfile):
#     filename=os.listdir(matpath)
#     numpy_data=np.transpose(trainsample)
#     #np.save('numpy_data',numpy_data)
#     for f in filename：
#     np.save(imgfile+"\\"+str(f))
