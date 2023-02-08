import numpy as np
import tensorflow as tf


# import cupy as np
# from numba import jit


def calculate_Similarity(data, n):
    Similarity = []
    point=[]
    # data1 = np.array([data[0, :], data[1, :]])
    data1 = np.array([data[8, :], data[9, :]])
    data2 = np.array([data[0, :], data[1, :]])
    if data[0,:] is not None:
        mean = np.mean(data[0,:])
    # n = len(data[8, :])
    # print('nnnnnnnnnnnnnn:',n)
    # N = int()
    # print('NNNNNNNNNNNNNNNNNNn:',N)
    # data_mean = 0.4 *( (max(data[0, :]) - min(data[0, :])) + (max(data[1, :]) - min(data[1, :])))
        mean_std = 0.5*np.abs(np.std(data[0, :]) - np.std(data[1, :]))
    else:
        mean =30
        mean_std = 20
    if mean_std != 0 :
        data_mean = (max(data[0, :]) - min(data[0, :]) + max(data[1, :]) - min(data[1, :]))/mean_std
    else:
        data_mean = (max(data[0, :]) - min(data[0, :]) + max(data[1, :]) - min(data[1, :]))/5
    # print('data_mean',data_mean)
    # data_mean_m = 0.08 * ((max(data[8, :]) - min(data[8, :])) + (max(data[9, :]) - min(data[9, :])))
    # print('mmmmmmmmmmdata_mean_m',data_mean_m)


    for i, j in enumerate(data[1, :]):
        RCS = np.sqrt(np.abs(data[5, i]) / np.pi)
        # RCS = np.abs(data[5, i])
        # print('RCS:',RCS)

        if i == 0:
            similarity = data_mean
            Similarity.append(similarity)
            point.append(0.5)


        else:
            a = [data_mean]
            P = [0.5]
            for p in range(1, i + 1):
                if np.abs(data2[1, i] - data2[1, i - p]) < RCS:
                    if np.abs(data2[0, i] - data2[0, i - p]) + np.abs(data2[1, i] - data2[1, i - p]) < data_mean:
                        similarity = np.sqrt(np.sum(np.square(data1[:, i] - data1[:, i - p])))
                        a.append(similarity)
                        P.append(data2[1, i] - data2[1, i - p])

            po = max(P)
            point.append(po)
            # print('11111111similarity:', a)
            Sim = min(a)
            Similarity.append(Sim)





    data1 = np.concatenate((data, [Similarity],[point]), axis=0)
    # mask = (data1[-1, :] != data_mean )
    # data1 = data1[:,mask]

    return data1, data_mean





def CBC(radar_data):
    # print('radar_data:', radar_data.shape)
    # lt = np.zeros((24, 1))
    # rt = np.zeros((24, 1))
    data = radar_data.T[radar_data.T[:, 1].argsort()].T
    # data = radar_data.T[np.argsort(-radar_data.T[:, 1])].T
    # print('data:',data)
    # data = np.transpose(data, (1, 0))
    N = len(data[1, :])
    radar_data, l_mean = calculate_Similarity(data, N)

    m = radar_data[-2, :]
    mask = (m[:] != l_mean)
    m = m[mask]
    # mask = (m[:] != r_mean)
    # m = m[mask]
    if m is not None:
        radar_mean = np.mean(m)
        radar_std = np.std(m)

    # print('len(m)',len(m))
        p = np.sort(m)   
        a = radar_mean - 1 * radar_std / np.sqrt(len(m))
        mask = (radar_data[-2, :] > a)
        radar_data1 = radar_data[:, mask]
        if radar_data1.size !=0:
            radar_data = radar_data1
        else:
            radar_data = radar_data
            
    else:
        radar_data = radar_data
 
    return radar_data




