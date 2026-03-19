

import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import transform

def myshuffle(train_scan_paths,train_labels,train_topo_paths):
    print("before shuffle")
    state = np.random.get_state()
    np.random.shuffle(train_scan_paths)
    np.random.set_state(state)
    np.random.shuffle(train_labels)
    np.random.set_state(state)
    np.random.shuffle(train_topo_paths)
    return train_scan_paths,train_labels,train_topo_paths
 


def resample(img_obj,sub_dir):###把路径输进去
    spacing = np.round(img_obj.GetSpacing(),decimals=4)
    origin = img_obj.GetOrigin()
    direction = img_obj.GetDirection()
    input_size = img_obj.GetSize()##输入的size应该都是一样的
    output_spacing = [1, 1, 1]
    output_size = [int(np.round((input_size[i] * spacing[i])) / output_spacing[i]) for i in range(3)]
    transform = sitk.Transform(3, sitk.sitkIdentity) #what's mean?
    ##如果是ROI  AP sitk.sitkNearestNeighbor
    interp = sitk.sitkNearestNeighbor if "PVP" or'AP' or'HBP' in sub_dir else sitk.sitkLinear  #是Im是线性插值，是label是最近邻插值
    resampled_img = sitk.Resample(img_obj, output_size, transform,
        interp, origin, output_spacing, direction)
    return resampled_img
##计算d,h,w，如果出现小于0或大于边界的情况，进行纠正
def getdhw(x0,x1,maxn,x):##maxn为最后输出的大小，0-x是边界，x0x1是病灶区域大小
    
    x_temp= (maxn - (x1-x0))/2##计算左右需要扩展多少
    if x0-x_temp<0:
        left = 0
        right = maxn
    elif x1+x_temp > x:
        left = x-maxn
        right = x
    else:
        left = x0-x_temp
        right = x1+x_temp
    if (maxn - (x1-x0))%2 != 0:##如果有0.5情况，进行四舍五入
        left = (int)(left+0.5)
        right = (int)(right+0.5)
    left = (int)(left)
    right = (int)(right)
    return left,right
def GetRoi2(sample_name, sz=128):##输入图片路径 输出固定大小的图片 若大于100则resize 否则外扩取
    # print("samplename",sample_name)
    img = sitk.ReadImage(sample_name)
    # mask_name = r'/home/pst/HCC_DATA/AP-ROI'##如果mask路径位置改变，记得改
    ##找到对应的ROI
    mask_name = r'/data/pst/hbp/HBP-ROI/HBP-ROI'
    sample_ROI = mask_name+'/'+sample_name[38:-3]
    # print("sample_ROI",sample_ROI)##得到ROI路径
    ##先取再重采样
    roi = sitk.ReadImage(sample_ROI)
    # img = resample(img,sample_name)
    # roi = resample(roi,sample_ROI)
    img_array = sitk.GetArrayFromImage(img)
    roi_array = sitk.GetArrayFromImage(roi)
    D, H, W = img_array.shape 

    ##裁剪/得到中间切片的位置，然后扩展成3D
    d_idx, h_idx, w_idx = np.where(roi_array > 0)###将大于0的区域保存下来
    d = list(set(d_idx))[1]##得到中间切片的位置
    array2d = roi_array[d]
    h_idx, w_idx = np.where(array2d > 0)
    h0, h1 = h_idx.min(), h_idx.max() 
    w0, w1 = w_idx.min(), w_idx.max() 
    d0, d1 = max(0, d-5 ), min(D, d+5 )##找到病灶区域最大最小的地方，确定立方体位置  并且稍微扩大一点
    h0, h1 = max(0, h0), min(H, h1 )##!!如果要更改外扩范围，得改这里
    w0, w1 = max(0, w0), min(W, w1 )
  
    img_crop = img_array[d0:d1, h0:h1, w0:w1]

    # ##统一到 同一尺寸
    # ##PadOrResize
    # if h1-h0 >sz or w1-w0 >sz:
    #     img_crop_resizeorpad = transform.resize(img_crop,(10,sz,sz))
    # else:
    #     h = h1-h0
    #     w = w1-w0
    #     x1 = (int)((sz - w)/2)
    #     x2 = sz-x1-w
    #     y1 = (int)((sz - h)/2)
    #     y2 = sz-y1-h

        
    #     img_crop_resizeorpad = np.pad(img_crop, ((0,0), (y1, y2), (x1, x2)), 'constant')
    ##直接resize
    img_crop_resizeorpad = transform.resize(img_crop,(10,sz,sz))


    return img_crop_resizeorpad
def GetRoi3(sample_name,  sz=100):##输入图片路径 输出固定大小的图片 若大于100则resize 否则外扩取
    print("samplename",sample_name)
    img = sitk.ReadImage(sample_name)
    mask_name = r'/home/pst/HCC_DATA/AP-ROI'##如果mask路径位置改变，记得改
 
    sample_ROI = mask_name+'/'+sample_name[30:-9]+'-AP.nii'##！！！
    print("sample_ROI",sample_ROI)##得到ROI路径

    roi = sitk.ReadImage(sample_ROI)
    img = resample(img,sample_name)
    roi = resample(roi,sample_ROI)
    img_array = sitk.GetArrayFromImage(img)
    roi_array = sitk.GetArrayFromImage(roi)
    D, H, W = img_array.shape 
    d_idx, h_idx, w_idx = np.where(roi_array > 0)###将大于0的区域保存下来
    d0, d1 = d_idx.min(), d_idx.max() 
    h0, h1 = h_idx.min(), h_idx.max() 
    w0, w1 = w_idx.min(), w_idx.max() 
    print("ROI区域",d0,d1,h0,h1,w0,w1)
    # ##margin
    # d0, d1 = max(0, (int)(0.9*d0) ), min(D, (int)(1.1*d1) )##找到病灶区域最大最小的地方，确定立方体位置  并且稍微扩大一点
    # h0, h1 = max(0, (int)(0.9*h0) ), min(H, (int)(1.1*h1) )##!!如果要更改外扩范围，得改这里
    # w0, w1 = max(0, (int)(0.9*w0) ), min(W, (int)(1.1*w1) )
    #margin
    LD = 0.2*(d1-d0)
    LH = 0.2*(h1-h0)
    LW = 0.2*(w1-w0)
    d0, d1 = max(0, (int)(d0 - LD) ), min(D, (int)(d1 + LD) )##找到病灶区域最大最小的地方，确定立方体位置  并且稍微扩大一点
    h0, h1 = max(0, (int)(h0 - LH) ), min(H, (int)(h1 + LH) )##!!如果要更改外扩范围，得改这里
    w0, w1 = max(0, (int)(w0 - LW) ), min(W, (int)(w1 + LW) )
    print("外扩后的结果",d0,d1,h0,h1,w0,w1)
    # ##
    # if d1-d0 >sz or h1-h0 >sz or w1-w0 >sz:
    #     img_crop = img_array[d0:d1,h0:h1,w0:w1]
    #     img_crop = transform.resize(img_crop,(sz,sz,sz))
    # else:
    #     ##统一到80
    #     d0,d1 = getdhw(d0,d1,sz,D)
    #     h0,h1 = getdhw(h0,h1,sz,H)
    #     w0,w1 = getdhw(w0,w1,sz,W)
    #     img_crop = img_array[d0:d1,h0:h1,w0:w1]
    #     print("调整后的结果的",d0,d1,h0,h1,w0,w1)
    img_crop = img_array[d0:d1, h0:h1, w0:w1]
    ##PadOrResize
    if d1-d0 >sz or h1-h0 >sz or w1-w0 >sz:
        img_crop_resizeorpad = transform.resize(img_crop,(sz,sz,sz))
    else:
        d = d1-d0
        h = h1-h0
        w = w1-w0
        x1 = (int)((sz - w)/2)
        x2 = sz-x1-w
        y1 = (int)((sz - h)/2)
        y2 = sz-y1-h
        z1 = (int)((sz - d)/2)
        z2 = sz-z1-d
        
        img_crop_resizeorpad = np.pad(img_crop, ((z1, z2), (y1, y2), (x1, x2)), 'constant')

    return img_crop_resizeorpad
def GetRoi(sample_name,  sz=100):##输入图片路径 输出固定大小的图片 若大于100则resize 否则外扩取
    print("samplename",sample_name)
    img = sitk.ReadImage(sample_name)
    # mask_name = r'/home/pst/HCC_DATA/AP-ROI'##如果mask路径位置改变，记得改
    mask_name = r'/data/pst/PVP-ROI'
    sample_ROI = mask_name+'/'+sample_name[22:-6]+'-PVP.nii'##！！！
    # print("sample_ROI",sample_ROI)##得到ROI路径

    roi = sitk.ReadImage(sample_ROI)
    img = resample(img,sample_name)
    roi = resample(roi,sample_ROI)
    img_array = sitk.GetArrayFromImage(img)
    roi_array = sitk.GetArrayFromImage(roi)
    D, H, W = img_array.shape 
    d_idx, h_idx, w_idx = np.where(roi_array > 0)###将大于0的区域保存下来
    d0, d1 = d_idx.min(), d_idx.max() 
    h0, h1 = h_idx.min(), h_idx.max() 
    w0, w1 = w_idx.min(), w_idx.max() 
    # print("ROI区域",d0,d1,h0,h1,w0,w1)
    # ##margin
    # d0, d1 = max(0, (int)(0.9*d0) ), min(D, (int)(1.1*d1) )##找到病灶区域最大最小的地方，确定立方体位置  并且稍微扩大一点
    # h0, h1 = max(0, (int)(0.9*h0) ), min(H, (int)(1.1*h1) )##!!如果要更改外扩范围，得改这里
    # w0, w1 = max(0, (int)(0.9*w0) ), min(W, (int)(1.1*w1) )
    #margin
    LD = 0.2*(d1-d0)
    LH = 0.2*(h1-h0)
    LW = 0.2*(w1-w0)
    d0, d1 = max(0, (int)(d0 - LD) ), min(D, (int)(d1 + LD) )##找到病灶区域最大最小的地方，确定立方体位置  并且稍微扩大一点
    h0, h1 = max(0, (int)(h0 - LH) ), min(H, (int)(h1 + LH) )##!!如果要更改外扩范围，得改这里
    w0, w1 = max(0, (int)(w0 - LW) ), min(W, (int)(w1 + LW) )
    # print("外扩后的结果",d0,d1,h0,h1,w0,w1)
    # ##
    # if d1-d0 >sz or h1-h0 >sz or w1-w0 >sz:
    #     img_crop = img_array[d0:d1,h0:h1,w0:w1]
    #     img_crop = transform.resize(img_crop,(sz,sz,sz))
    # else:
    #     ##统一到80
    #     d0,d1 = getdhw(d0,d1,sz,D)
    #     h0,h1 = getdhw(h0,h1,sz,H)
    #     w0,w1 = getdhw(w0,w1,sz,W)
    #     img_crop = img_array[d0:d1,h0:h1,w0:w1]
    #     print("调整后的结果的",d0,d1,h0,h1,w0,w1)
    img_crop = img_array[d0:d1, h0:h1, w0:w1]
    ##PadOrResize
    if d1-d0 >sz or h1-h0 >sz or w1-w0 >sz:
        img_crop_resizeorpad = transform.resize(img_crop,(sz,sz,sz))
    else:
        d = d1-d0
        h = h1-h0
        w = w1-w0
        x1 = (int)((sz - w)/2)
        x2 = sz-x1-w
        y1 = (int)((sz - h)/2)
        y2 = sz-y1-h
        z1 = (int)((sz - d)/2)
        z2 = sz-z1-d
        
        img_crop_resizeorpad = np.pad(img_crop, ((z1, z2), (y1, y2), (x1, x2)), 'constant')

    return img_crop_resizeorpad
def GetRoi_topo_pvp(sample_name,  sz=100):##输入图片路径 输出固定大小的图片 若大于100则resize 否则外扩取
    print("samplename",sample_name)
    img = sitk.ReadImage(sample_name)

    mask_name = '/data/pst/'+sample_name[10:14]+'ROI'
    sample_ROI = mask_name+'/'+sample_name[22:-6]+'-'+sample_name[10:13]+'.nii'##！！！
    print("sample_ROI",sample_ROI)##得到ROI路径
    roi = sitk.ReadImage(sample_ROI)

    img = resample(img,sample_name)
    roi = resample(roi,sample_ROI)
    img_array = sitk.GetArrayFromImage(img)
    roi_array = sitk.GetArrayFromImage(roi)

    # ##save resample img
    
    # print("save image")
    # img_obj = sitk.GetImageFromArray(img_array)
    # sitk.WriteImage(img_obj,'img.nii.gz')
    # roi_obj = sitk.GetImageFromArray(roi_array)
    # sitk.WriteImage(roi_obj,'roi.nii.gz')
    D, H, W = img_array.shape 
    d_idx, h_idx, w_idx = np.where(roi_array > 0)###将大于0的区域保存下来
    d0, d1 = d_idx.min(), d_idx.max() 
    h0, h1 = h_idx.min(), h_idx.max() 
    w0, w1 = w_idx.min(), w_idx.max() 
    print("ROI区域",d0,d1,h0,h1,w0,w1)
    # ##margin
    # d0, d1 = max(0, (int)(0.9*d0) ), min(D, (int)(1.1*d1) )##找到病灶区域最大最小的地方，确定立方体位置  并且稍微扩大一点
    # h0, h1 = max(0, (int)(0.9*h0) ), min(H, (int)(1.1*h1) )##!!如果要更改外扩范围，得改这里
    # w0, w1 = max(0, (int)(0.9*w0) ), min(W, (int)(1.1*w1) )
    #margin
    LD = 0.2*(d1-d0)
    LH = 0.2*(h1-h0)
    LW = 0.2*(w1-w0)
    d0, d1 = max(0, (int)(d0 - LD) ), min(D, (int)(d1 + LD) )##找到病灶区域最大最小的地方，确定立方体位置  并且稍微扩大一点
    h0, h1 = max(0, (int)(h0 - LH) ), min(H, (int)(h1 + LH) )##!!如果要更改外扩范围，得改这里
    w0, w1 = max(0, (int)(w0 - LW) ), min(W, (int)(w1 + LW) )
    print("DHW",D,H,W)
    print("外扩后的结果",d0,d1,h0,h1,w0,w1)
    # ##
    # if d1-d0 >sz or h1-h0 >sz or w1-w0 >sz:
    #     img_crop = img_array[d0:d1,h0:h1,w0:w1]
    #     img_crop = transform.resize(img_crop,(sz,sz,sz))
    # else:
    #     ##统一到80
    #     d0,d1 = getdhw(d0,d1,sz,D)
    #     h0,h1 = getdhw(h0,h1,sz,H)
    #     w0,w1 = getdhw(w0,w1,sz,W)
    #     img_crop = img_array[d0:d1,h0:h1,w0:w1]
    #     print("调整后的结果的",d0,d1,h0,h1,w0,w1)
    img_crop = img_array[d0:d1, h0:h1, w0:w1]
    # obj = sitk.GetImageFromArray(img_crop)
    # sitk.WriteImage(obj,'crop.nii.gz')

    return img_crop
def GetRoi_topo(sample_name,  sz=100):##输入图片路径 输出固定大小的图片 若大于100则resize 否则外扩取
    print("samplename",sample_name)
    img = sitk.ReadImage(sample_name)

    mask_name = '/data/pst/'+sample_name[10:14]+'ROI'
    sample_ROI = mask_name+'/'+sample_name[22:-9]+'-'+sample_name[10:13]+'.nii'##！！！
    print("sample_ROI",sample_ROI)##得到ROI路径
    roi = sitk.ReadImage(sample_ROI)

    img = resample(img,sample_name)
    roi = resample(roi,sample_ROI)
    img_array = sitk.GetArrayFromImage(img)
    roi_array = sitk.GetArrayFromImage(roi)

    # ##save resample img
    
    # print("save image")
    # img_obj = sitk.GetImageFromArray(img_array)
    # sitk.WriteImage(img_obj,'img.nii.gz')
    # roi_obj = sitk.GetImageFromArray(roi_array)
    # sit