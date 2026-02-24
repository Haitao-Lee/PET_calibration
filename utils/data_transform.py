# coding = utf - 8
import numpy as np
from config import args
from tqdm import tqdm
import data_input
import os
import random
import struct


def regular_y(y, num_eps=8):
    y_norm = np.linalg.norm(y, axis=1)
    idx = np.argmin(y_norm).flatten()[0]
    reg_y = np.zeros(y.shape)
    reg_y[0, :] = y[idx, :]
    y[idx, :] = y[idx, :] + 512
    for i in range(1, 256):
        if i % 16 == 0:
            sum_y = np.sum(y, axis=1)
            idx = np.argmin(sum_y)
            reg_y[i, :] = y[idx, :]
            y[idx, :] = y[idx, :] + 512
            continue        
        differ = y - np.expand_dims(reg_y[i-1], 0).repeat(256, axis=0)
        differ_norm = np.linalg.norm(differ, axis=1)
        candidate_dist = np.sort(differ_norm, axis=0)[:num_eps]
        indices = []
        eval = []
        for j in range(num_eps):
            tmp_idx = np.argwhere(differ_norm==candidate_dist[j])[0][0]
            indices.append(tmp_idx)
            tmp_differ = differ[tmp_idx, :]
            tmp_differ = tmp_differ/candidate_dist[j]
            if tmp_differ[1] < 0.5 and tmp_differ[0] > 0 or candidate_dist[j] > 50:
                eval.append(0)
            else:
            # if i > 16:
            #     y[tmp_idx,:] - reg_y[i-16,:]
                eval.append((tmp_differ[1]-0.3*tmp_differ[0])/candidate_dist[j])
        idx = np.argmax(np.array(eval))
        idx = indices[idx]
        reg_y[i, :] = y[idx, :]
        y[idx, :] = y[idx, :] + 512
    return reg_y

def test_regular_y(y, alpha, theta, num_eps=5):
    y_norm = np.linalg.norm(y, axis=1)
    idx = np.argmin(y_norm).flatten()[0]
    reg_y = np.zeros(y.shape)
    reg_y[0, :] = y[idx, :]
    
    y[idx, :] = y[idx, :] + 512
    for i in range(1, 256):
        if i % 16 == 0:
            sum_y = np.sum(y, axis=1)
            idx = np.argmin(sum_y)
            reg_y[i, :] = y[idx, :]
            y[idx, :] = y[idx, :] + 512
            continue        
        differ = y - np.expand_dims(reg_y[i-1], 0).repeat(256, axis=0)
        differ_norm = np.linalg.norm(differ, axis=1)
        candidate_dist = np.sort(differ_norm, axis=0)[:num_eps]
        indices = []
        eval = []
        for j in range(num_eps):
            tmp_idx = np.argwhere(differ_norm==candidate_dist[j])[0][0]
            indices.append(tmp_idx)
            tmp_differ = differ[tmp_idx, :]
            tmp_differ = tmp_differ/candidate_dist[j]
            # a = np.cos(theta)
            # print(a)
            if  tmp_differ[1] < np.cos(theta) and tmp_differ[0] > 0  or candidate_dist[j] > 50:
                eval.append(0)
            else:
            # if i > 16:
            #     y[tmp_idx,:] - reg_y[i-16,:]
                eval.append((alpha*tmp_differ[1]-(1-alpha)*tmp_differ[0])/candidate_dist[j])
        idx = np.argmax(np.array(eval))
        idx = indices[idx]
        reg_y[i, :] = y[idx, :]
        y[idx, :] = y[idx, :] + 512
    effect = True
    for q in range(16):
        if reg_y[q*16, 1] > 50 or reg_y[q*16 + 15, 1] < 200 or reg_y.max() > 255:
            effect = False
            break
    return reg_y, effect



def data_reg_reg(luts, peaks_x, peaks_y):
    tmp_peaks_luts = np.zeros(luts.shape)
    effect_luts = []
    effect_peaks = []
    # peak_size = 5
    for i in tqdm(range((tmp_peaks_luts.shape[0])), desc='\033[31mreloadng:\033[0m'):
        for j in range(peaks_x[i].shape[0]):
            for k in range(peaks_y[i].shape[1]):
                # tmp_peaks_luts[i][min(max(0, peaks_x[i][j][k]-peak_size//2), 1023):min(max(0, peaks_x[i][j][k]+peak_size//2), 1023), min(max(0, peaks_y[i][j][k]-peak_size//2), 1023):min(max(0, peaks_y[i][j][k]+peak_size//2), 1023)] = 0.2*luts[i].max()
                tmp_peaks_luts[i][min(max(0, peaks_x[i][j][k]), 1023), min(max(0, peaks_y[i][j][k]), 1023)] = luts[i][min(max(0, peaks_x[i][j][k]), 1023), min(max(0, peaks_y[i][j][k]), 1023)]
    for i in tqdm(range((tmp_peaks_luts.shape[0])), desc='\033[31mtransforming:\033[0m'):
        for j in range(4):
            for k in range(4):
                single_lut = luts[i][j*256:(j+1)*256, k*256:(k+1)*256]
                single_peak = tmp_peaks_luts[i][j*256:(j+1)*256, k*256:(k+1)*256]
                if single_lut.mean() != 0:
                    tmp_effect_peaks = []
                    for m in range(single_lut.shape[0]):
                        for n in range(single_lut.shape[1]):
                            if single_peak[m, n] != 0:
                                if m > 50 and m < 200 and n > 50 and n < 200 and single_lut[m, n] < single_lut.mean():
                                    continue
                                tmp_effect_peaks.append(np.array([m,n]))
                    if len(tmp_effect_peaks) != 256:
                        continue
                    tmp_reg_peaks = regular_y(np.array(tmp_effect_peaks))
                    effect = True
                    for q in range(16):
                        if tmp_reg_peaks[q*16, 1] > 50 or tmp_reg_peaks[q*16 + 15, 1] < 200 or tmp_reg_peaks.max() > 255:
                            effect = False
                            break
                    if effect:
                        effect_peaks.append(tmp_reg_peaks)
                        effect_luts.append(single_lut)
    effect_luts = np.array(effect_luts)
    effect_peaks = np.array(effect_peaks)
    return effect_luts, effect_peaks


def data_reg_seg(luts, peaks_x, peaks_y):
    tmp_peaks_luts = np.zeros(luts.shape)
    effect_luts = []
    effect_peaks = []
    peak_size = 5
    for i in tqdm(range((tmp_peaks_luts.shape[0])), desc='\033[31mreloadng:\033[0m'):
        for j in range(peaks_x[i].shape[0]):
            for k in range(peaks_y[i].shape[1]):
                tmp_peaks_luts[i][min(max(0, peaks_x[i][j][k]-peak_size//2), 1023):min(max(0, peaks_x[i][j][k]+peak_size//2), 1023), min(max(0, peaks_y[i][j][k]-peak_size//2), 1023):min(max(0, peaks_y[i][j][k]+peak_size//2), 1023)] = 0.2*luts[i].max()
                # tmp_peaks_luts[i][min(max(0, peaks_x[i][j][k]), 1023), min(max(0, peaks_y[i][j][k]), 1023)] = 0.2*luts[i].max()
    for i in tqdm(range((tmp_peaks_luts.shape[0])), desc='\033[31mtransforming:\033[0m'):
        for j in range(4):
            for k in range(4):
                tmp_lut = np.array(luts[i, j*256:(j+1)*256, k*256:(k+1)*256])
                if tmp_lut.max() != 0:
                    effect_luts.append(tmp_lut)
                    effect_peaks.append(np.array(tmp_peaks_luts[i, j*256:(j+1)*256, k*256:(k+1)*256]))
    effect_luts = np.array(effect_luts)
    effect_peaks = np.array(effect_peaks)
    return effect_luts, effect_peaks


def raw2npy(img_paths, xp_paths, yp_paths):
    for path in tqdm(img_paths, desc="\033[31mLoading images:\033[0m"):
        with open(path, 'rb') as raw_lut:
            size = os.path.getsize(path)
            lut = [struct.unpack('>I', raw_lut.read(4))[0] for _ in range(size // 4)]
            lut = np.array(lut).reshape([1024, 1024])
            file_name = os.path.splitext(os.path.basename(path))[0]







def main(args=args):
    lut_names, peak_X_names, peak_Y_names = data_input.get_data_file_names(args.data_dir, args.origin_lut, args.origin_peak)
    peaks_x, peaks_y = data_input.load_origin_peaks(peak_X_names, peak_Y_names)
    luts = data_input.load_origin_luts(lut_names)
    ef_luts, ef_peaks = data_reg_reg(luts, peaks_x, peaks_y)
    if not os.path.exists(args.reg_peaks_npy + '/training/'):
        os.makedirs(args.reg_peaks_npy + '/training/')
    if not os.path.exists(args.reg_peaks_npy + '/validation/'):
        os.makedirs(args.reg_peaks_npy + '/validation/')
    if not os.path.exists(args.reg_luts_npy + '/training/'):
        os.makedirs(args.reg_luts_npy + '/training/')
    if not os.path.exists(args.reg_luts_npy + '/validation/'):
        os.makedirs(args.reg_luts_npy + '/validation/')
    val_idx = random.sample(range(ef_luts.shape[0]), round(ef_luts.shape[0]*args.data_seg[1]))
    for i in range(ef_luts.shape[0]):
        if i in val_idx:
            np.save(args.reg_luts_npy + '/validation/lut%d.npy' % i, ef_luts[i, :, :])
            np.save(args.reg_peaks_npy + '/validation/peak%d.npy' % i, ef_peaks[i, :, :])
        else:
            np.save(args.reg_luts_npy + '/training/lut%d.npy' % i, ef_luts[i, :, :])
            np.save(args.reg_peaks_npy + '/training/peak%d.npy' % i, ef_peaks[i, :, :])
    return 0


# if __name__ == '__main__':
#     # main()
#     alphas = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1])
#     thetas = np.pi*np.array([1/4,1/3,5/12,1/2])
#     _, peaks = data_input.load_validation_data()
#     for n in range(alphas.shape[0]):
#         for m in range(thetas.shape[0]):
#             rate = 0
#             for i in range(0,100):
#                 _, res = test_regular_y(peaks[i+50].copy(), alphas[n], thetas[m])
#                 if res == True:
#                     rate = rate + 0.01
#             print('%s:%s, %s:%s , rate:%s' %(r'$\alpha$',str(alphas[n]), r'$\theta$',str(thetas[m]), str(rate)))   
    
