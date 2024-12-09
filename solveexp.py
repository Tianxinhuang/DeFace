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
    #print(np.shape(landmarks0),np.shape(landmarks))
    loss = torch.norm(landmarks0 - landmarks, 2, dim=-1).pow(2)
    #print(loss.shape)
    #assert False
    #.sum(-1)
    loss = loss.mean()+loss.max(1)[0].mean()
    return loss
def regStatModel(coeff, var):
    loss = ((coeff * coeff) / var).mean(-1).mean()
    return loss
def runStep1(shapeMean,idx,bfm09,id0,exps,box):
    #print("1/3 => Optimizing head pose and expressions using landmarks...", file=sys.stderr, flush=True)
    torch.set_grad_enabled(True)
    min09,max09,min17,max17=box
    exp17,bs09=exps
    shapeMean=torch.tensor(shapeMean).to('cuda')
    bfm09=torch.tensor(bfm09).to('cuda')
    min09=torch.tensor(min09).to('cuda').unsqueeze(0)
    max09=torch.tensor(max09).to('cuda').unsqueeze(0)
    min17=torch.tensor(min17).to('cuda').unsqueeze(0)
    max17=torch.tensor(max17).to('cuda').unsqueeze(0)

    exp17=torch.tensor(exp17).to('cuda')
    bs09=torch.tensor(bs09).to('cuda')

    params = [
        #{'params': pipeline.vRotation, 'lr': 0.02},
        #{'params': pipeline.vTranslation, 'lr': 0.02},
        {'params':  pipeline.vExpCoeff, 'lr': 0.005},
        #{'params': pipeline.vShapeCoeff, 'lr': 0.02}
    ]
    pipeline.vExpCoeff.requires_grad = True

    #if self.config.optimizeFocalLength:
    #    params.append({'params': self.pipeline.vFocals, 'lr': 0.02})

    optimizer = torch.optim.Adam(params)
    losses = []
    vertices = shapeMean
    for iter in range(10000):
    #for iter in tqdm.tqdm(range(self.config.iterStep1)):
        optimizer.zero_grad()
        #vertices = pipeline.computeShape()
        vertices0 = shapeMean.unsqueeze(0) + torch.einsum('ij,aj->ai', exp17.reshape([-1,100]), pipeline.vExpCoeff).reshape([46,-1,3])
        #vertices = vertices.reshape([46,-1,3])
        #+ torch.einsum('ni,ijk->njk', (pipeline.vShapeCoeff, pipeline.morphableModel.shapePca)).squeeze(0)
        vertices = (max09-min09)*vertices0/(max17-min17)+min09-min17*(max09-min09)/(max17-min17)
        vert09 = bfm09.unsqueeze(0) + torch.einsum('ij,aj->ai', bs09, torch.eye(46,dtype=torch.double).to('cuda')).reshape([46,-1,3])#46*N*3
        landmarks=vert09[:,id0]
        #print(shapeMean.shape,pipeline.morphableModel.shapePca.shape,vertices.shape)
        #assert False
        #cameraVertices = pipeline.transformVertices(vertices)
        loss = landmarkLoss(vertices[:,idx], landmarks)
        #loss += 0.001 * regStatModel(pipeline.vExpCoeff, pipeline.morphableModel.expressionPcaVar)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        #print(iter, '=>', loss.item())
    print(vertices[0,idx],landmarks[0])
    print(vertices[1,idx],landmarks[1])
    nexp17=vertices0-shapeMean.unsqueeze(0)
    return nexp17.detach().cpu().numpy()

    #self.plotLoss(losses, 0, self.outputDir + 'checkpoints/stage1_loss.png')
    #self.saveParameters(self.outputDir + 'checkpoints/stage1_output.pickle')

if __name__=='__main__':
    dict = loadDictionaryFromPickle('/apdcephfs/private_yaoshihuang/CPEM/NextFace2/baselMorphableModel/morphableModel-2017.pickle')
    pathLandmarks = '/apdcephfs/private_yaoshihuang/CPEM/NextFace2/baselMorphableModel/landmark_62.txt'
    bfm17 = dict['shapeMean']
    shapevar=dict['shapePcaVar']
    exp17 = dict['expressionPca']
    bfm17 = loadmat('/apdcephfs/private_yaoshihuang/CPEM/new_bfm17.mat')['id']
    expp17=loadmat('/apdcephfs/private_yaoshihuang/CPEM/new_bfm17.mat')['exp']
    #print(np.shape(expp17))
    #assert False
    #deltaB = self.deltaB.reshape([self.expnum,-1]).transpose(0,1)
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
    bs09=np.load('/apdcephfs/private_yaoshihuang/CPEM/data/BFM/mean_delta_blendshape.npy')
    #bs09=np.reshape(bs09,[46,-1,3])

    id1=np.loadtxt(pathLandmarks, delimiter='\t\t')[:, 1].astype(np.int64)
    id0=np.loadtxt(pathLandmarks, delimiter='\t\t')[:, 0].astype(np.int64)
    id09=(model['keypoints']-1).astype(int)[0]
    id9=id09[id0]
    land9 = bfm09[id9]
    #print((model['keypoints'] - 1).astype(int).shape,id0)
    #land9=land09[id0]
    land17=bfm17[id1]

    cen09,r09=cenr(bfm09)
    cen17,r17=cenr(bfm17)
    #nbfm17=(bfm17-cen17)*r09/r17+cen09
    min17,max17=box(land17)
    min09,max09=box(land9)
    nbfm17=(max09-min09)*bfm17/(max17-min17)+min09-min17*(max09-min09)/(max17-min17)
    pipeline.initSceneParameters(46, False)
    nbfm17=torch.tensor(nbfm17).to('cuda')
    nexp17=runStep1(bfm17,id1,bfm09,id9,[exp17,bs09],[min09,max09,min17,max17])
    #print(np.shape(nexp17))
    #nbfm17=nbfm17.detach().cpu().numpy()
    #bfm=(nbfm17-min09)*(max17-min17)/(max09-min09)+min17

    savemat('/apdcephfs/private_yaoshihuang/CPEM/new_bfm17.mat',{'id':bfm17,'exp':nexp17})
    #bfms=np.concatenate([nbfm17+np.array([3,0,0]),bfm09],axis=0)
    #pcd=o3d.geometry.PointCloud()
    #pcd.points=o3d.utility.Vector3dVector(bfms[:,:3])
    #o3d.io.write_point_cloud('bfms.pcd', pcd, write_ascii=True, compressed=False, print_progress=False) 
