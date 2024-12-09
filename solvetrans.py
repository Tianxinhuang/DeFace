#!/usr/bin/env python
# coding=utf-8
import numpy as np
#import open3d as o3d
from scipy.io import loadmat, savemat
import pickle5 as pickle
import torch
import sys
sys.path.append('/apdcephfs/private_yaoshihuang/CPEM/NextFace/')
from pipeline import Pipeline
from config import Config
config=Config()
pipeline = Pipeline(config)
def loadDictionaryFromPickle(picklePath):
    handle = open(picklePath, 'rb')
    assert handle is not None
    dic = pickle.load(handle)
    handle.close()
    return dic
#myshape:n*3
def cenr(myshape):
    cen=np.mean(myshape,axis=0,keepdims=True)
    r=np.max(np.sqrt(np.sum(np.square(myshape-cen),axis=-1)))
    return cen,r
def box(myshape):
    minv=np.min(myshape,axis=0,keepdims=True)
    maxv=np.max(myshape,axis=0,keepdims=True)
    return minv, maxv
def landmarkLoss(landmarks0, landmarks):
    print(np.shape(landmarks0),np.shape(landmarks))
    loss = torch.norm(landmarks0 - landmarks, 2, dim=-1).pow(2).mean()
    return loss
def regStatModel(coeff, var):
    loss = ((coeff * coeff) / var).mean(-1).mean()
    return loss
def runStep1(shapeMean,idx,landmarks):
    #print("1/3 => Optimizing head pose and expressions using landmarks...", file=sys.stderr, flush=True)
    torch.set_grad_enabled(True)

    params = [
        #{'params': pipeline.vRotation, 'lr': 0.02},
        #{'params': pipeline.vTranslation, 'lr': 0.02},
        #{'params': self.pipeline.vExpCoeff, 'lr': 0.02},
        {'params': pipeline.vShapeCoeff, 'lr': 0.02}
    ]
    pipeline.vShapeCoeff.requires_grad = True

    #if self.config.optimizeFocalLength:
    #    params.append({'params': self.pipeline.vFocals, 'lr': 0.02})

    optimizer = torch.optim.Adam(params)
    losses = []
    vertices = shapeMean
    for iter in range(3000):
    #for iter in tqdm.tqdm(range(self.config.iterStep1)):
        optimizer.zero_grad()
        #vertices = pipeline.computeShape()
        vertices = shapeMean + torch.einsum('ni,ijk->njk', (pipeline.vShapeCoeff, pipeline.morphableModel.shapePca)).squeeze(0)
        #print(shapeMean.shape,pipeline.morphableModel.shapePca.shape,vertices.shape)
        #assert False
        #cameraVertices = pipeline.transformVertices(vertices)
        loss = landmarkLoss(vertices[idx], torch.tensor(landmarks).to('cuda'))
        loss += 0.1 * regStatModel(pipeline.vShapeCoeff, pipeline.morphableModel.shapePcaVar)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        #print(iter, '=>', loss.item())
    print(vertices[idx],landmarks)
    return vertices

    #self.plotLoss(losses, 0, self.outputDir + 'checkpoints/stage1_loss.png')
    #self.saveParameters(self.outputDir + 'checkpoints/stage1_output.pickle')

if __name__=='__main__':
    dict = loadDictionaryFromPickle('/apdcephfs/private_yaoshihuang/CPEM/NextFace2/baselMorphableModel/morphableModel-2017.pickle')
    pathLandmarks = '/apdcephfs/private_yaoshihuang/CPEM/NextFace2/baselMorphableModel/landmark_62.txt'
    bfm17 = dict['shapeMean']
    shapevar=dict['shapePcaVar']
    #print(np.shape(shapevar))
    #assert False
    #cen17,r17=cenr(bfm17)
    #rotlist=np.reshape(np.array([np.pi,0,0]),[1,3])
    #rotation = self.Compute_rotation_matrix(torch.from_numpy(rotlist)).to(device)
    
    #rotshape=torch.matmul(self.meanshape, rotation)
    model = loadmat('/apdcephfs/private_yaoshihuang/CPEM/data/BFM/BFM_model_front.mat')
        # mean face shape. [1, N*3]
    bfm09mean = model['meanshape']
    bfm09 = np.reshape(bfm09mean,[-1,3])
    id1=np.loadtxt(pathLandmarks, delimiter='\t\t')[:, 1].astype(np.int64)
    id0=np.loadtxt(pathLandmarks, delimiter='\t\t')[:, 0].astype(np.int64)
    land09 = bfm09[(model['keypoints']-1).astype(int)[0]]
    #print((model['keypoints'] - 1).astype(int).shape,id0)
    land09=land09[id0]
    land17=bfm17[id1]

    cen09,r09=cenr(bfm09)
    cen17,r17=cenr(bfm17)
    #nbfm17=(bfm17-cen17)*r09/r17+cen09
    min17,max17=box(land17)
    min09,max09=box(land09)
    nbfm17=(max09-min09)*bfm17/(max17-min17)+min09-min17*(max09-min09)/(max17-min17)
    pipeline.initSceneParameters(1, False)
    nbfm17=torch.tensor(nbfm17).to('cuda')
    nbfm17=runStep1(nbfm17,id1,land09)
    nbfm17=nbfm17.detach().cpu().numpy()
    bfm=(nbfm17-min09)*(max17-min17)/(max09-min09)+min17
    assert False

    savemat('/apdcephfs/private_yaoshihuang/CPEM/new_bfm17.mat',{'id':bfm})
    #bfms=np.concatenate([nbfm17+np.array([3,0,0]),bfm09],axis=0)
    #pcd=o3d.geometry.PointCloud()
    #pcd.points=o3d.utility.Vector3dVector(bfms[:,:3])
    #o3d.io.write_point_cloud('bfms.pcd', pcd, write_ascii=True, compressed=False, print_progress=False) 
