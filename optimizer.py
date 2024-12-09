from image import Image, ImageFolder, overlayImage, saveImage
from gaussiansmoothing import GaussianSmoothing, smoothImage
from projection import estimateCameraPosition

from textureloss import TextureLoss
from pipeline import Pipeline
from config import Config
from utils2 import *
import argparse
import pickle
import tqdm
import sys
import torch

sys.path.append('faceparsing')
from segmask import pic2mask2
import numpy as np
import random
from facenet_pytorch import InceptionResnetV1

class Optimizer:

    def __init__(self, outputDir, config):
        torch.manual_seed(1024) #
        torch.cuda.manual_seed(1024) #
        np.random.seed(1024)
        random.seed(1024)
        torch.backends.cudnn.deterministic = True

        self.config = config
        self.device = config.device
        self.verbose = config.verbose
        self.framesNumber = 0
        self.pipeline = Pipeline(self.config)

        if self.config.lamdmarksDetectorType == 'fan':
            from landmarksfan import LandmarksDetectorFAN
            self.landmarksDetector = LandmarksDetectorFAN(self.pipeline.morphableModel.landmarksMask, self.device)
        elif self.config.lamdmarksDetectorType == 'mediapipe':
            from landmarksmediapipe import LandmarksDetectorMediapipe
            self.landmarksDetector = LandmarksDetectorMediapipe(self.pipeline.morphableModel.landmarksMask, self.device)
        else:
            raise ValueError(f'lamdmarksDetectorType must be one of [mediapipe, fan] but was {self.config.lamdmarksDetectorType}')

        self.textureLoss = TextureLoss(self.device)

        self.inputImage = None
        self.landmarks = None
        torch.set_grad_enabled(False)
        self.smoothing = GaussianSmoothing(3, 3, 1.0, 2).to(self.device)
        self.outputDir = outputDir + '/'
        self.debugDir = self.outputDir + '/debug/'
        mkdir_p(self.outputDir)

        self.vEnhancedDiffuse = None
        self.vEnhancedSpecular = None
        self.vEnhancedRoughness = None

        #facenet for human loss
        self.FaceNet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
        # fix parameters
        for param in self.FaceNet.parameters():
            param.requires_grad = False

        self.idx = torch.tensor(list(range(self.pipeline.cnum)),dtype=torch.int32).to(self.device)

    #Human prior constraint
    def human_loss(self, img):
        self.FaceNet.classify=True
        pred_em = self.FaceNet(img)
        prob_em = torch.softmax(pred_em, dim = 1)
        result = (torch.ones_like(prob_em)-prob_em).square().min(1)[0]
        result = result.mean()
        return result

   #inpics:n*w*h*c
   #kmeans for the hue
    def kmeans(self, inpics):
        from sklearn.cluster import KMeans
        from PIL import Image
        from scipy.cluster.vq import vq, kmeans, whiten

        X = inpics.cpu().numpy()#.reshape([-1,3])
        X=X.reshape([-1,3])

        kcluster = KMeans(n_clusters=16,init='random', max_iter=100000, random_state=0)
        cluster = kcluster.fit(X)
        cens = cluster.cluster_centers_
        cens = torch.tensor(cens).to(self.device)#/255
        return cens

    #Global prior constraint
    def global_loss(self,s3img,s2img,csize=3,cens=None):
        s3img=s3img.permute(0,3,1,2)
        s2img=s2img.permute(0,3,1,2)
        isize=s3img.shape[2]//csize
        bsize=s3img.shape[0]
        import torch
        if cens is not None:
            cens=cens.permute(1,0)#cens.reshape([3,-1])
            t2img=cens[None,:,:,None,None].repeat([bsize,1,1,s3img.shape[2],s3img.shape[2]])
        else:
            t2img=torch.nn.Unfold(kernel_size=(csize,csize),padding=csize//2, stride=1)(s2img)
            t2img=t2img.reshape(bsize,3,csize*csize,s3img.shape[2],s3img.shape[2])#.contiguous()

        t3img=s3img.unsqueeze(2)

        diff=(t3img-t2img).square().sum(1,keepdim=True).min(2)[0]#/3.0
        mindiff=diff.mean()
        return mindiff

    def softdis2(self,s3img,s2img,csize=7):
        s3img=s3img.permute(0,3,1,2)
        s2img=s2img.permute(0,3,1,2)
        isize=s3img.shape[2]//csize
        bsize=s3img.shape[0]
        t2img=torch.nn.Unfold(kernel_size=(csize,csize),padding=csize//2, stride=1)(s2img)
        #print(t2img.shape)
        t2img=t2img.reshape(bsize,3,csize*csize,s3img.shape[2],s3img.shape[2])
        t3img=s3img.unsqueeze(2)
        result=t3img-t2img
        return result

    #Local prior constraint
    def local_loss(self, refimage, s3):
        refimage = refimage[...,:3]
        idiff = ((self.softdis2(refimage, refimage, csize=5)-self.softdis2(s3,s3,csize=5)).square()).sum(1).sum(2,keepdim=True).permute(0,2,3,1)
        return idiff

    def saveParameters(self, outputFileName):

        dict = {
            'vShapeCoeff': self.pipeline.vShapeCoeff.detach().cpu().numpy(),
            'vAlbedoCoeff': self.pipeline.vAlbedoCoeff.detach().cpu().numpy(),
            'vExpCoeff': self.pipeline.vExpCoeff.detach().cpu().numpy(),
            'vRotation': self.pipeline.vRotation.detach().cpu().numpy(),
            'vTranslation': self.pipeline.vTranslation.detach().cpu().numpy(),
            'vFocals': self.pipeline.vFocals.detach().cpu().numpy(),
            'vShCoeffs': self.pipeline.vShCoeffsn.detach().cpu().numpy(),
            'screenWidth':self.pipeline.renderer.screenWidth,
            'screenHeight': self.pipeline.renderer.screenHeight,
            'sharedIdentity': self.pipeline.sharedIdentity,
            'outmask': self.mlamda,
            'light_mask': self.light_masks,
            'content_mask': self.mlamda,
            'light_idx': self.idx,
        }

        if self.vEnhancedDiffuse is not None:
            dict['vEnhancedDiffuse'] = self.vEnhancedDiffuse.detach().cpu().numpy()
        if self.vEnhancedSpecular is not None:
            dict['vEnhancedSpecular'] = self.vEnhancedSpecular.detach().cpu().numpy()
        if self.vEnhancedRoughness is not None:
            dict['vEnhancedRoughness'] = self.vEnhancedRoughness.detach().cpu().numpy()

        handle = open(outputFileName, 'wb')
        pickle.dump(dict, handle, pickle.HIGHEST_PROTOCOL)
        handle.close()

    def loadParameters(self, pickelFileName):
        handle = open(pickelFileName, 'rb')
        assert handle is not None
        dict = pickle.load(handle)
        self.pipeline.vShapeCoeff = torch.tensor(dict['vShapeCoeff']).to(self.device)
        self.pipeline.vAlbedoCoeff = torch.tensor(dict['vAlbedoCoeff']).to(self.device)
        self.pipeline.vExpCoeff = torch.tensor(dict['vExpCoeff']).to(self.device)
        self.pipeline.vRotation = torch.tensor(dict['vRotation']).to(self.device)
        self.pipeline.vTranslation = torch.tensor(dict['vTranslation']).to(self.device)
        self.pipeline.vFocals = torch.tensor(dict['vFocals']).to(self.device)
        self.pipeline.vShCoeffsn = torch.tensor(dict['vShCoeffs']).to(self.device)
        self.pipeline.renderer.screenWidth = int(dict['screenWidth'])
        self.pipeline.renderer.screenHeight = int(dict['screenHeight'])
        self.pipeline.sharedIdentity = bool(dict['sharedIdentity'])
        self.mlamda = torch.tensor(dict['content_mask']).to(self.device)
        self.light_masks = torch.tensor(dict['light_mask']).to(self.device)
        self.idx = torch.tensor(dict['light_idx']).to(self.device)

        if "vEnhancedDiffuse" in dict:
            self.vEnhancedDiffuse = torch.tensor(dict['vEnhancedDiffuse']).to(self.device)

        if "vEnhancedSpecular" in dict:
            self.vEnhancedSpecular = torch.tensor(dict['vEnhancedSpecular']).to(self.device)

        if "vEnhancedRoughness" in dict:
            self.vEnhancedRoughness = torch.tensor(dict['vEnhancedRoughness']).to(self.device)

        handle.close()
        self.enableGrad()

    def enableGrad(self):
        self.pipeline.vShapeCoeff.requires_grad = True
        self.pipeline.vAlbedoCoeff.requires_grad = True
        self.pipeline.vExpCoeff.requires_grad = True
        self.pipeline.vRotation.requires_grad = True
        self.pipeline.vTranslation.requires_grad = True
        self.pipeline.vFocals.requires_grad = True
        #self.pipeline.vShCoeffs.requires_grad = True
        #self.pipeline.vShCoeffs2.requires_grad = True
        self.pipeline.vShCoeffsn.requires_grad = True
        #self.pipeline.maskcen.requires_grad = True
        #self.pipeline.maskr.requires_grad = True
        #self.pipeline.maskl.requires_grad = True
        #self.pipeline.tao.requires_grad = True



    def setImage(self, imagePath, sharedIdentity = False):
        '''
        set image to estimate face reflectance and geometry
        :param imagePath: drive path to the image
        :param sharedIdentity: if true than the shape and albedo coeffs are equal to 1, as they belong to the same person identity
        :return:
        '''
        if os.path.isfile(imagePath):
            self.inputImage = Image(imagePath, self.device, self.config.maxResolution)
        else:
            self.inputImage = ImageFolder(imagePath, self.device, self.config.maxResolution)

        self.framesNumber = self.inputImage.tensor.shape[0]
        self.pipeline.renderer.screenWidth = self.inputImage.width
        self.pipeline.renderer.screenHeight = self.inputImage.height

        print('detecting landmarks using:', self.config.lamdmarksDetectorType)
        landmarks = self.landmarksDetector.detect(self.inputImage.tensor)
        #assert (landmarks.shape[0] == 1)  # can only handle single subject in image
        assert (landmarks.dim() == 3 and landmarks.shape[2] == 2)
        self.landmarks = landmarks
        for i in range(self.framesNumber):
            imagesLandmark = self.landmarksDetector.drawLandmarks(self.inputImage.tensor[i], self.landmarks[i])
            cv2.imwrite(self.outputDir  + '/landmarks' + str(i) + '.png', cv2.cvtColor(imagesLandmark, cv2.COLOR_BGR2RGB) )
        self.pipeline.initSceneParameters(self.framesNumber, sharedIdentity)
        self.enableGrad()
        self.initCameraPos() #always init the head pose (rotation + translation)

    def initCameraPos(self):
        print('init camera pose...', file=sys.stderr, flush=True)
        association = self.pipeline.morphableModel.landmarksAssociation
        vertices = self.pipeline.computeShape()
        headPoints = vertices[:, association]
        rot, trans = estimateCameraPosition(self.pipeline.vFocals, self.inputImage.center,
                                    self.landmarks, headPoints, self.pipeline.vRotation,
                                    self.pipeline.vTranslation)

        self.pipeline.vRotation = rot.clone().detach()
        self.pipeline.vTranslation = trans.clone().detach()
    def getTextureIndex(self, i):
        if self.pipeline.sharedIdentity:
            return 0
        return i
    def debugFrame(self, image, target, diffuseTexture, specularTexture, roughnessTexture, outputPrefix):
        for i in range(image.shape[0]):
            diff = (image[i] - target[i]).abs()

            import cv2
            diffuse = cv2.resize(cv2.cvtColor(diffuseTexture[self.getTextureIndex(i)].detach().cpu().numpy(), cv2.COLOR_BGR2RGB), (target.shape[2], target.shape[1]))
            spec = cv2.resize(cv2.cvtColor(specularTexture[self.getTextureIndex(i)].detach().cpu().numpy(), cv2.COLOR_BGR2RGB),  (target.shape[2], target.shape[1]))
            rough = roughnessTexture[self.getTextureIndex(i)].detach().cpu().numpy()
            rough = cv2.cvtColor(cv2.resize(rough, (target.shape[2], target.shape[1])), cv2.COLOR_GRAY2RGB)

            res = cv2.hconcat([cv2.cvtColor(image[i].detach().cpu().numpy(), cv2.COLOR_BGR2RGB),
                               cv2.cvtColor(target[i].detach().cpu().numpy(), cv2.COLOR_BGR2RGB),
                               cv2.cvtColor(diff.detach().cpu().numpy(), cv2.COLOR_BGR2RGB)])
            ref = cv2.hconcat([diffuse, spec, rough])

            debugFrame = cv2.vconcat([np.power(np.clip(res, 0.0, 1.0), 1.0 / 2.2) * 255, ref * 255])
            cv2.imwrite(outputPrefix  + '_frame' + str(i) + '.png', debugFrame)

    def regStatModel(self, coeff, var):
        loss = ((coeff * coeff) / var).mean()
        return loss

    def plotLoss(self, lossArr, index, fileName):
        import matplotlib.pyplot as plt
        plt.figure(index)
        plt.plot(lossArr)
        plt.scatter(np.arange(0, len(lossArr)).tolist(), lossArr, c='red')
        plt.savefig(fileName)

    def landmarkLoss(self, cameraVertices, landmarks):
        return self.pipeline.landmarkLoss(cameraVertices, landmarks, self.pipeline.vFocals, self.inputImage.center)

    #Same as NextFace to optimize 3DMM geometrical parameters
    def runStep1(self):
        print("1/3 => Optimizing head pose and expressions using landmarks...", file=sys.stderr, flush=True)
        torch.set_grad_enabled(True)

        params = [
            {'params': self.pipeline.vRotation, 'lr': 0.02},
            {'params': self.pipeline.vTranslation, 'lr': 0.02},
            {'params': self.pipeline.vExpCoeff, 'lr': 0.02},
        ]

        if self.config.optimizeFocalLength:
            params.append({'params': self.pipeline.vFocals, 'lr': 0.02})

        optimizer = torch.optim.Adam(params)
        losses = []

        for iter in tqdm.tqdm(range(self.config.iterStep1)):
            optimizer.zero_grad()
            vertices = self.pipeline.computeShape()
            cameraVertices = self.pipeline.transformVertices(vertices)
            loss = self.landmarkLoss(cameraVertices, self.landmarks)
            loss += 0.1 * self.regStatModel(self.pipeline.vExpCoeff, self.pipeline.morphableModel.expressionPcaVar)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if self.verbose:
                print(iter, '=>', loss.item())

    #Optimize all 3DMM parameters, f(\cdot), g{\cdot}
    def runStep2(self):
        print("2/3 => Optimizing shape, statistical albedos, expression, head pose and scene light...", file=sys.stderr, flush=True)
        torch.set_grad_enabled(True)
        self.pipeline.renderer.samples = 8
        inputTensor = torch.pow(self.inputImage.tensor, self.inputImage.gamma)

        optimizer = torch.optim.Adam([
            {'params': self.pipeline.vShCoeffsn, 'lr': self.config.gamma_lr},
            {'params': self.pipeline.folding.parameters(), 'lr': 0.007},
            {'params': self.pipeline.maskfunc.parameters(), 'lr': 0.007},
            {'params': self.pipeline.vShapeCoeff, 'lr': 0.01},
            {'params': self.pipeline.vAlbedoCoeff, 'lr': 0.007},
            {'params': self.pipeline.vExpCoeff, 'lr': 0.01},
            {'params': self.pipeline.vRotation, 'lr': 0.0001},
            {'params': self.pipeline.vTranslation, 'lr': 0.0001}
        ])
        losses = []
        iteri = 0

        #Weights for $L_{area}$ and $L_{bin}$
        lw=self.config.w3

        for iter in tqdm.tqdm(range(self.config.iterStep2 + 1)):
            optimizer.zero_grad()
            vertices, diffAlbedo, specAlbedo = self.pipeline.morphableModel.computeShapeAlbedo(self.pipeline.vShapeCoeff, self.pipeline.vExpCoeff, self.pipeline.vAlbedoCoeff)
            cameraVerts = self.pipeline.camera.transformVertices(vertices, self.pipeline.vTranslation, self.pipeline.vRotation)
            diffuseTextures = self.pipeline.morphableModel.generateTextureFromAlbedo(diffAlbedo)
            specularTextures = self.pipeline.morphableModel.generateTextureFromAlbedo(specAlbedo)

            #$f(\cdot)$
            selfmasks,maskl = self.pipeline.ftlayers()

            imagelist = []
            #Render faces under different lighting conditions
            for i in range(self.idx.shape[0]):
                imagelist.append(self.pipeline.render(cameraVerts, diffuseTextures, specularTextures, shcoeffs=self.pipeline.vShCoeffsn[:,self.idx[i]]).unsqueeze(1))
            imagelist=torch.cat(imagelist,dim=1)

            #Render mask $M_R$
            mask = imagelist[:,0,:,:, 3:]
           
            #Do the ACE to select effective light masks after iter0
            if iteri == self.config.iterStep0:
                problist = (selfmasks * mask.permute(0,3,1,2)).sum([0,2,3])/mask.permute(0,3,1,2).sum()
                vals, idx = torch.sort(problist,dim=0,descending=True)
                print(vals)
                sumi = 0
                lw = self.config.w4
                for i in range(self.pipeline.cnum):
                    sumi+=vals[i]
                    if vals[i]>self.config.epsilon:
                        self.idx = idx[:i+1]
                imagelist = imagelist[:,list(self.idx.cpu())]

            cmask1=selfmasks[:,list(self.idx.cpu())]
            cmask1 = cmask1/cmask1.sum(1).unsqueeze(1)
            images = (cmask1.unsqueeze(-1)*imagelist).sum(1)

            smoothedImage = smoothImage(images[..., 0:3].float(), self.smoothing)

            #Get the results from pre-trained face region prediction network 
            if iteri==0:
                self.fmask = pic2mask2(inputTensor, self.landmarks.cpu().numpy()).detach().clone()

            #$g(\cdot)$
            self.fmask2 = self.pipeline.getfmask()
            
            #face region
            lamda = mask * self.fmask2#

            #Occlusion mask $M_o$ to get unoccluded face regions for next stage 
            self.mlamda=lamda.detach().clone()            
            
            #Combining the rendered face and surroundings
            smoothedImage = (1-lamda) * inputTensor + lamda * smoothedImage

            #Loss to distillation loss $L_{seg}$
            floss = self.config.w2*(lamda-self.fmask).square().sum(dim=[1,2]) #video 200 
            #floss = 300 * (lamda-self.fmask).square().sum(dim=[1,2])
            floss = floss / self.fmask.sum(dim=[1,2])
            floss = floss.mean()

            #Photometric loss
            diff = 2.0 * mask * ((smoothedImage - inputTensor).abs())
            photoLoss = self.config.w0 * diff.mean() 

            cmean = cmask1 * mask.permute(0,3,1,2)
            cmean2 = cmean.sum(dim=[-1,-2],keepdim=True)/mask.permute(0,3,1,2).sum(dim=[-1,-2],keepdim=True)
            cmean = cmask1.mean(dim=1,keepdim=True)

            #$L_{area}$
            maskrloss  = 1 *((-(cmask1-cmean).square()).exp() * mask.permute(0,3,1,2)).sum()/mask.sum()-1
            #$L_{bin}$
            maskrloss2 = 1 *((-(cmask1-cmean2).square()).exp() * mask.permute(0,3,1,2)).sum()/mask.sum()-1

            if iteri <= 100:
                maskrloss = maskrloss
            else:
                maskrloss = maskrloss2
 
            #Same losses as NextFace
            landmarksLoss = self.config.w1 * self.config.weightLandmarksLossStep2 *  self.landmarkLoss(cameraVerts, self.landmarks)
            regLoss = 0.0001 * self.pipeline.vShCoeffsn.pow(2).mean()#+0.0001 * self.pipeline.vShCoeffs2.pow(2).mean()
            regLoss += 1.0 * self.regStatModel(self.pipeline.vAlbedoCoeff, self.pipeline.morphableModel.diffuseAlbedoPcaVar)
            regLoss += self.config.weightShapeReg * self.regStatModel(self.pipeline.vShapeCoeff, self.pipeline.morphableModel.shapePcaVar)
            regLoss += self.config.weightExpressionReg * self.regStatModel(self.pipeline.vExpCoeff, self.pipeline.morphableModel.expressionPcaVar)

            loss = photoLoss + landmarksLoss + regLoss
            loss += floss #$L_{seg}$
            loss += lw * maskrloss #$L_{area}$ or $L_{bin}$

            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            iteri += 1

            if self.verbose:
                print(iter, ' => Loss:', loss.item(),
                      '. photo Loss:', photoLoss.item(),
                      '. landmarks Loss: ', landmarksLoss.item(),
                      '. regLoss: ', regLoss.item(),
                      '. maskLoss: ', maskrloss.item(),)


        self.vEnhancedDiffuse = diffuseTextures.detach().clone()
        self.vEnhancedSpecular = specularTextures.detach().clone()
        self.vEnhancedRoughness = self.pipeline.vRoughness.detach().clone() if self.vEnhancedRoughness is None else self.vEnhancedRoughness.detach().clone()

    def gradient(self, inputs, outputs, create_graph=True, retain_graph=True):
        d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
        points_grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=d_points,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]#[:, -3:]
        return points_grad

    #Tuning the texture T, f(\cdot), lighting conditions
    def runStep3(self):
        print("3/3 => finetuning albedos, shape, expression, head pose and scene light...", file=sys.stderr, flush=True)
        torch.set_grad_enabled(True)
        self.pipeline.renderer.samples = 8

        inputTensor = torch.pow(self.inputImage.tensor, self.inputImage.gamma)
        vertices, diffAlbedo, specAlbedo = self.pipeline.morphableModel.computeShapeAlbedo(self.pipeline.vShapeCoeff, self.pipeline.vExpCoeff, self.pipeline.vAlbedoCoeff) 

        vDiffTextures = self.pipeline.morphableModel.generateTextureFromAlbedo(diffAlbedo).detach().clone() if self.vEnhancedDiffuse is None else self.vEnhancedDiffuse.detach().clone()
        vSpecTextures = self.pipeline.morphableModel.generateTextureFromAlbedo(specAlbedo).detach().clone() if self.vEnhancedSpecular is None else self.vEnhancedSpecular.detach().clone()
        vRoughTextures = self.pipeline.vRoughness.detach().clone() if self.vEnhancedRoughness is None else self.vEnhancedRoughness.detach().clone()

        refDiffTextures = vDiffTextures.detach().clone()
        refSpecTextures = vSpecTextures.detach().clone()
        refRoughTextures = vRoughTextures.detach().clone()
        vDiffTextures.requires_grad = True
        vSpecTextures.requires_grad = True
        vRoughTextures.requires_grad = True

        #Hue calculated with K-means
        cens = self.kmeans(refDiffTextures)

        optimizer = torch.optim.Adam([
            {'params': vDiffTextures, 'lr': 0.005},
            {'params': vSpecTextures, 'lr': 0.005},
            {'params': vRoughTextures, 'lr': 0.01},
            {'params': self.pipeline.folding.parameters(), 'lr': 0.001},
            {'params': self.pipeline.vShCoeffsn, 'lr': 0.001},
        ])
        losses = []

        #Mask to fiter the gradients not in the texture regions
        a,b,c,d=vDiffTextures.shape
        gmask = torch.ones([a,b,c,1], dtype=torch.float32, device=self.device)
        rmask = torch.where(vDiffTextures.sum(-1,keepdim=True)>1e-5,gmask,torch.zeros_like(gmask))

        refdiffuseAlbedo = self.pipeline.render(diffuseTextures=refDiffTextures, renderAlbedo=True)
        refspecularAlbedo = self.pipeline.render(diffuseTextures=refSpecTextures, renderAlbedo=True)
        refroughnessAlbedo = self.pipeline.render(diffuseTextures=refRoughTextures.repeat(1, 1, 1, 3), renderAlbedo=True)

        for iter in tqdm.tqdm(range(self.config.iterStep3 + 1)):
            optimizer.zero_grad()
            vertices, diffAlbedo, specAlbedo = self.pipeline.morphableModel.computeShapeAlbedo(self.pipeline.vShapeCoeff, self.pipeline.vExpCoeff, self.pipeline.vAlbedoCoeff)
            cameraVerts = self.pipeline.camera.transformVertices(vertices, self.pipeline.vTranslation, self.pipeline.vRotation)
            
            diffuseAlbedo = self.pipeline.render(diffuseTextures=vDiffTextures, renderAlbedo=True)
            specularAlbedo = self.pipeline.render(diffuseTextures=vSpecTextures, renderAlbedo=True)
            roughnessAlbedo = self.pipeline.render(diffuseTextures=vRoughTextures.repeat(1, 1, 1, 3), renderAlbedo=True)

            blist=[]
            
            #GP Loss
            loss_global = self.global_loss(vDiffTextures,refDiffTextures,13, cens)

            #HP Loss calculated under multiple lighting conditions
            imagelist=[]
            for i in range(self.idx.shape[0]):
                img_ref = self.pipeline.render(cameraVerts, vDiffTextures, vSpecTextures, vRoughTextures, shcoeffs=self.pipeline.vShCoeffsn[:,self.idx[i]])
                imagelist.append(img_ref.unsqueeze(1))
                loss_idi = self.human_loss((img_ref[...,:3]).float().permute(0,3,1,2))
                blist.append(loss_idi)
            imagelist=torch.cat(imagelist,dim=1)
            loss_hp = sum(blist)/len(blist)

            #Masks for different lighting conditions
            selfmasks,maskl = self.pipeline.ftlayers()

            #Use lighting masks to combine face images
            vcmask11=selfmasks[:,list(self.idx.cpu())]
            vcmask11 = vcmask11/vcmask11.sum(1).unsqueeze(1)
            images = (vcmask11.unsqueeze(-1)*imagelist).sum(1)

            #Use render mask to get the face region
            mask = imagelist[:,0,:,:, 3:]
            self.light_masks = (vcmask11 * mask.permute(0,3,1,2)).unsqueeze(-1)
            smoothedImage = smoothImage(images[..., 0:3].float(), self.smoothing)
            s0=smoothedImage

            #Occlusion mask is not updated anymore in Stage 3
            lamda = self.mlamda

            #Get rendered smoothed faces and textures in the face space
            smoothedImage = (1-lamda) * inputTensor + lamda * smoothedImage
            diffuseAlbedo = (diffuseAlbedo * (1-lamda) + diffuseAlbedo.detach().clone() * lamda)[...,:3]
            specularAlbedo = (specularAlbedo * (1-lamda) + specularAlbedo.detach().clone() * lamda)[...,:3]
            roughnessAlbedo = (roughnessAlbedo * (1-lamda) + roughnessAlbedo.detach().clone() * lamda)[...,:3]

            #Calculate the $LP Loss$
            idiff = mask * (5*self.local_loss(refdiffuseAlbedo, diffuseAlbedo) + 6*self.local_loss(refspecularAlbedo, specularAlbedo) + self.local_loss(refroughnessAlbedo, roughnessAlbedo))
            loss_local = (idiff.sum(dim=[1,2])/mask.sum(dim=[1,2])).mean()

            #calculate the $L_{bin}$
            cmean = vcmask11 * mask.permute(0,3,1,2)
            cmean = cmean.sum(dim=[-1,-2],keepdim=True)/mask.permute(0,3,1,2).sum(dim=[-1,-2],keepdim=True)
            maskrloss = ((-(vcmask11-cmean).square()).exp() * mask.permute(0,3,1,2)).sum()/mask.sum() -1

            #Same constraints as NextFace
            diff =  mask * (smoothedImage - inputTensor).abs()
            loss = 2 * self.config.w0 * diff.mean() 
            loss += 0.2 * (self.textureLoss.regTextures(vDiffTextures, refDiffTextures, ws = self.config.weightDiffuseSymmetryReg, wr =  self.config.weightDiffuseConsistencyReg, wc = self.config.weightDiffuseConsistencyReg, wsm = self.config.weightDiffuseSmoothnessReg, wm = 0.) + \
                    self.textureLoss.regTextures(vSpecTextures, refSpecTextures, ws = self.config.weightSpecularSymmetryReg, wr = self.config.weightSpecularConsistencyReg, wc = self.config.weightSpecularConsistencyReg, wsm = self.config.weightSpecularSmoothnessReg, wm = 0.5) + \
                    self.textureLoss.regTextures(vRoughTextures, refRoughTextures, ws = self.config.weightRoughnessSymmetryReg, wr = self.config.weightRoughnessConsistencyReg, wc = self.config.weightRoughnessConsistencyReg, wsm = self.config.weightRoughnessSmoothnessReg, wm = 0.))
            loss += 0.0001 * self.pipeline.vShCoeffsn.pow(2).mean()
            loss += self.config.weightExpressionReg * self.regStatModel(self.pipeline.vExpCoeff, self.pipeline.morphableModel.expressionPcaVar)
            loss += self.config.weightShapeReg * self.regStatModel(self.pipeline.vShapeCoeff, self.pipeline.morphableModel.shapePcaVar)
            loss += self.config.weightLandmarksLossStep3 * self.landmarkLoss(cameraVerts, self.landmarks)

            loss += 5.0 * maskrloss #$L_{bin}$
            loss += self.config.w7 * loss_hp #HP Loss
            loss += self.config.w6 * loss_local #LP Loss
            loss += self.config.w5 * loss_global #GP Loss

            losses.append(loss.item())
            loss.backward(retain_graph=True)

            #Reduce the noises beyond the texture regions by remove possible gradients
            vDiffTextures.grad *= rmask
            vSpecTextures.grad *= rmask
            optimizer.step()

            if self.verbose:
                print(iter, ' => Loss:', loss.item())

        self.vEnhancedDiffuse = vDiffTextures.detach().clone()
        self.vEnhancedSpecular = vSpecTextures.detach().clone()
        self.vEnhancedRoughness = vRoughTextures.detach().clone()

    #Save pictures including occlusion masks, light masks, rendered faces, textures...
    def savepics(self, samples,  outputDir = None, prefix = ''):
        if outputDir is None:
            outputDir = self.outputDir
            mkdir_p(outputDir)

        self.namelist=[]

        if len(self.inputImage.imageNames)==0:
            self.namelist.append(self.inputImage.imageName)
        else:
            self.namelist=self.inputImage.imageNames

        print("saving to: '", outputDir, "'. hold on... ", file=sys.stderr, flush=True)
        outputDir += '/' #use join

        inputTensor = torch.pow(self.inputImage.tensor, self.inputImage.gamma)
        vDiffTextures = self.vEnhancedDiffuse
        vSpecTextures = self.vEnhancedSpecular
        vRoughTextures = self.vEnhancedRoughness
        vertices, diffAlbedo, specAlbedo = self.pipeline.morphableModel.computeShapeAlbedo(self.pipeline.vShapeCoeff, self.pipeline.vExpCoeff, self.pipeline.vAlbedoCoeff)
        cameraVerts = self.pipeline.camera.transformVertices(vertices, self.pipeline.vTranslation, self.pipeline.vRotation)
        cameraNormals = self.pipeline.morphableModel.computeNormals(cameraVerts)


        if vDiffTextures is None:
            vDiffTextures = self.pipeline.morphableModel.generateTextureFromAlbedo(diffAlbedo)
            vSpecTextures = self.pipeline.morphableModel.generateTextureFromAlbedo(specAlbedo)
            vRoughTextures = self.pipeline.vRoughness

        self.pipeline.renderer.samples = samples

        imagelist=[]
        print_id = 0
        for i in range(self.idx.shape[0]):
            imagelist.append(self.pipeline.render(cameraVerts, vDiffTextures, vSpecTextures, vRoughTextures, shcoeffs=self.pipeline.vShCoeffsn[:,self.idx[i]]).unsqueeze(1))

            for print_id in range(len(self.namelist)):
                saveImage(imagelist[-1][print_id].squeeze(), outputDir + prefix + 'light_render'+str(print_id)+'_'+str(i)+'.png')
                saveImage(self.light_masks[print_id][i].repeat(1,1,3), outputDir + prefix + 'light_mask'+str(print_id)+'_'+str(i)+'.png')

        imagelist=torch.cat(imagelist,dim=1)
        images = (self.light_masks*imagelist).sum(1)#[...,:3]

        mask = images[...,3:]
        images0 = images
        images = self.mlamda * images[...,:3] + (1-self.mlamda) * inputTensor[...,:3]
        images = torch.cat([images,mask],-1)

        occlusion = (1-self.mlamda)* mask * inputTensor[...,:3]
        face = self.mlamda * images0[...,:3]
        oface = mask * images0[...,:3]
        occlusion_mask = (1-self.mlamda) * mask
        face_mask = self.mlamda
        maskface = self.mlamda * inputTensor[...,:3]
        envir = (1-mask) * inputTensor[...,:3]
        fullmask = mask

        for i in range(len(self.namelist)):

            saveImage(face_mask[i].repeat(1,1,3), outputDir + prefix + 'occlu_face_mask_'+str(i)+'.png')
            saveImage(occlusion_mask[i].repeat(1,1,3), outputDir + prefix + 'occlu_outer_mask_'+str(i)+'.png')
            saveImage(face[i], outputDir + prefix + 'occlu_face_'+str(i)+'.png')
            saveImage(maskface[i], outputDir + prefix + 'mask_face_'+str(i)+'.png')
            saveImage(occlusion[i], outputDir + prefix + 'occlu_outer_'+str(i)+'.png')
            saveImage(envir[i], outputDir + prefix + 'envir_'+str(i)+'.png')
            saveImage(oface[i], outputDir + prefix + 'full_face_'+str(i)+'.png')
            saveImage(fullmask[i].repeat(1,1,3), outputDir + prefix + 'full_mask_'+str(i)+'.png')


            overlay = overlayImage(inputTensor[i], images0[i])
            saveImage(torch.cat([overlay.to(self.device), torch.ones_like(images[i])[..., 3:]], dim = -1), outputDir + self.namelist[i].split('.')[0] + '_face'+'.png')

            overlay = overlayImage(inputTensor[i], images[i])
            saveImage(torch.cat([overlay.to(self.device), torch.ones_like(images[i])[..., 3:]], dim = -1), outputDir + self.namelist[i].split('.')[0] + '_merge'+'.png')

        diffuseAlbedo = self.pipeline.render(diffuseTextures=vDiffTextures, renderAlbedo=True)
        specularAlbedo = self.pipeline.render(diffuseTextures=vSpecTextures, renderAlbedo=True)
        roughnessAlbedo = self.pipeline.render(diffuseTextures=vRoughTextures.repeat(1, 1, 1, 3), renderAlbedo=True)

        saveImage(diffuseAlbedo[self.getTextureIndex(i)], outputDir + prefix + 'diffuseMap.png')
        saveImage(specularAlbedo[self.getTextureIndex(i)], outputDir + prefix + 'specularMap.png')
        saveImage(roughnessAlbedo[self.getTextureIndex(i)], outputDir + prefix  + 'roughnessMap.png')
