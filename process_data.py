
# coding: utf-8
# this is process_data function.
# need to feed flag :'training','test','validation' as string
#return data dictionary 
#example:Data=process_data('training') ,then Red_input=Data['Red'] as a [batch_size,1080] array
# data in the directory '../MAE_KITTI/data_18x60/'

def process_data(flag):
    
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import scipy.io
    
    
    data_index=pd.read_csv('../MAE_KITTI/kitti_split.txt',header=None)
    
    if flag=='validation':
        input_index=data_index[59:61]
    elif flag=='training':
        input_index=data_index[1:29]
    elif flag=='test':
        input_index=data_index[31:58]
    else:
        print('error')
        
    foldernamme='../MAE_KITTI/data_18x60/'
    input_index=input_index.values
    xcr1=np.empty((1080,0))
    xcg1=np.empty((1080,0))
    xcb1=np.empty((1080,0))
    xid1=np.empty((1080,0))
    dmask1=np.empty((1080,0))
    Ground1=np.empty((1080,0))
    Objects1=np.empty((1080,0))
    Building1=np.empty((1080,0))
    Vegetation1=np.empty((1080,0))
    Sky1=np.empty((1080,0))
    xcr2=np.empty((1080,0))
    xcg2=np.empty((1080,0))
    xcb2=np.empty((1080,0))
    xid2=np.empty((1080,0))
    dmask2=np.empty((1080,0))
    Ground2=np.empty((1080,0))
    Objects2=np.empty((1080,0))
    Building2=np.empty((1080,0))
    Vegetation2=np.empty((1080,0))
    Sky2=np.empty((1080,0))
    
    
    
    for i in range(len(input_index)):
        
        name=input_index[i][0]
        RGB1name=foldernamme+'data_kitti_im02_'+name+'_18x60.mat'
        RGB2name=foldernamme+'data_kitti_im03_'+name+'_18x60.mat'
    
        Depth1name=foldernamme+'data_kitti_InvDepth02_'+name+'_18x60'
        Depth2name=foldernamme+'data_kitti_InvDepth03_'+name+'_18x60'
    
        Sem1name=foldernamme+'data_kitti_seg02_'+name+'_18x60'
        Sem2name=foldernamme+'data_kitti_seg03_'+name+'_18x60'

        mat1=scipy.io.loadmat(RGB1name)
        xcr1=np.append(xcr1,mat1['xcr'],axis=1)
        xcg1=np.append(xcg1,mat1['xcg'],axis=1)
        xcb1=np.append(xcb1,mat1['xcb'],axis=1)
    
        mat2=scipy.io.loadmat(RGB2name)
        xcr2=np.append(xcr2,mat2['xcr'],axis=1)
        xcg2=np.append(xcg2,mat2['xcg'],axis=1)
        xcb2=np.append(xcb2,mat2['xcb'],axis=1)
    
    
        matd1=scipy.io.loadmat(Depth1name)
        xid1=np.append(xid1,matd1['xid'],axis=1)
        dmask1=np.append(dmask1,matd1['xmask'],axis=1)
    
        matd2=scipy.io.loadmat(Depth2name)
        xid2=np.append(xid2,matd2['xid'],axis=1)
        dmask2=np.append(dmask2,matd1['xmask'],axis=1)
    
    

        mats1=scipy.io.loadmat(Sem1name)

        Ground1=np.append(Ground1,(mats1['xss']==1).astype(int),axis=1)
        Objects1=np.append(Objects1,(mats1['xss']==2).astype(int),axis=1)
        Building1=np.append(Building1,(mats1['xss']==3).astype(int),axis=1)
        Vegetation1=np.append(Vegetation1,(mats1['xss']==4).astype(int),axis=1)
        Sky1=np.append(Sky1,(mats1['xss']==5).astype(int),axis=1)
    
        mats2=scipy.io.loadmat(Sem2name)
        Ground2=np.append(Ground2,(mats2['xss']==1).astype(int),axis=1)
        Objects2=np.append(Objects2,(mats2['xss']==2).astype(int),axis=1)
        Building2=np.append(Building2,(mats2['xss']==3).astype(int),axis=1)
        Vegetation2=np.append(Vegetation2,(mats2['xss']==4).astype(int),axis=1)
        Sky2=np.append(Sky2,(mats2['xss']==5).astype(int),axis=1)
        
        
    xcr1=np.transpose(xcr1)
    xcg1=np.transpose(xcg1)
    xcb1=np.transpose(xcb1)
    xid1=np.transpose(xid1)
    dmask1=np.transpose(dmask1)
    Ground1=np.transpose(Ground1)
    Objects1=np.transpose(Objects1)
    Building1=np.transpose(Building1)
    Vegetation1=np.transpose(Vegetation1)
    Sky1=np.transpose(Sky1)

    xcr2=np.transpose(xcr2)
    xcg2=np.transpose(xcg2)
    xcb2=np.transpose(xcb2)
    xid2=np.transpose(xid2)
    dmask2=np.transpose(dmask2)
    Ground2=np.transpose(Ground2)
    Objects2=np.transpose(Objects2)
    Building2=np.transpose(Building2)
    Vegetation2=np.transpose(Vegetation2)
    Sky2=np.transpose(Sky2)
    

    Depth_data=np.concatenate((xid1,xid2),axis=0)
    Depthmask_data=np.concatenate((dmask1,dmask2),axis=0)
    Red_data=np.concatenate((xcr1,xcr2),axis=0)/255.0
    Green_data=np.concatenate((xcg1,xcg2),axis=0)/255.0
    Blue_data=np.concatenate((xcb1,xcb2),axis=0)/255.0
    Ground_data=np.concatenate((Ground1,Ground2),axis=0)
    Objects_data=np.concatenate((Objects1,Objects2),axis=0)
    Building_data=np.concatenate((Building1,Building2),axis=0)
    Vegetation_data=np.concatenate((Vegetation1,Vegetation2),axis=0)
    Sky_data=np.concatenate((Sky1,Sky2),axis=0)   
    
    return {'Red':Red_data,
            'Green':Green_data,
            'Blue':Blue_data,
            'Depth':Depth_data,
            'Depthmask':Depthmask_data,
            'Ground':Ground_data,
            'Objects':Objects_data,
            'Building':Building_data,
            'Vegetation':Vegetation_data,
            'Sky':Sky_data}





