from sphericalharmonics import SphericalHarmonics
from morphablemodel import MorphableModel
from renderer import Renderer
from camera import Camera
from utils2 import *
import torch.nn as nn

class Pipeline:

    def __init__(self,  config):
        '''
        a pipeline can generate and render textured faces under different camera angles and lighting conditions
        :param config: configuration file used to parameterize the pipeline
        '''
        self.cnum=config.lnum
        self.config = config
        self.device = config.device
        self.camera = Camera(self.device)
        self.sh = SphericalHarmonics(config.envMapRes, self.device)

        if self.config.lamdmarksDetectorType == 'fan':
            pathLandmarksAssociation = '/landmark_62.txt'
        elif self.config.lamdmarksDetectorType == 'mediapipe':
            pathLandmarksAssociation = '/landmark_62_mp.txt'
        else:
            raise ValueError(f'lamdmarksDetectorType must be one of [mediapipe, fan] but was {self.config.lamdmarksDetectorType}')

        self.morphableModel = MorphableModel(path = config.path,
                                             textureResolution= config.textureResolution,
                                             trimPca= config.trimPca,
                                             landmarksPathName=pathLandmarksAssociation,
                                             device = self.device
                                             )
        self.renderer = Renderer(config.rtTrainingSamples, 1, self.device)
        self.uvMap = self.morphableModel.uvMap.clone()
        self.uvMap[:, 1] = 1.0 - self.uvMap[:, 1]
        #self.faces32 = self.morphableModel.faces.to(torch.int32).contiguous()
        self.faces32 = torch.Tensor(np.loadtxt('newface.txt')).reshape(-1, 3).to(self.device).to(torch.int32).contiguous()
        self.shBands = config.bands
        self.sharedIdentity = config.shareid

    def initSceneParameters(self, n, sharedIdentity = False):
        '''
        init pipeline parameters (face shape, albedo, exp coeffs, light and  head pose (camera))
        :param n: the the number of parameters (if negative than the pipeline variables are not allocated)
        :param sharedIdentity: if true, the shape and albedo coeffs are equal to 1, as they belong to the same person identity
        :return:
        '''

        if n <= 0:
            return

        self.sharedIdentity = sharedIdentity
        nShape = 1 if sharedIdentity == True else n
        ngamma = n
        ngamma = 1 if sharedIdentity == True else n

        #self.cnum = 10#2

        self.vShapeCoeff = torch.zeros([nShape, self.morphableModel.shapeBasisSize], dtype = torch.float32, device = self.device)
        self.vAlbedoCoeff = torch.zeros([nShape, self.morphableModel.albedoBasisSize], dtype=torch.float32, device=self.device)

        self.vExpCoeff = torch.zeros([n, self.morphableModel.expBasisSize], dtype=torch.float32, device=self.device)
        self.vRotation = torch.zeros([n, 3], dtype=torch.float32, device=self.device)
        self.vTranslation = torch.zeros([n, 3], dtype=torch.float32, device=self.device)
        self.vTranslation[:, 2] = 500.
        self.vRotation[:, 0] = 3.14
        self.vFocals = self.config.camFocalLength * torch.ones([n], dtype=torch.float32, device=self.device)
        
        self.vShCoeffs = 0.0 * torch.ones([ngamma, self.shBands * self.shBands, 3], dtype=torch.float32, device=self.device)
        self.vShCoeffs[..., 0, 0] = 0.5
        self.vShCoeffs[..., 2, 0] = -0.5
        self.vShCoeffs[..., 1] = self.vShCoeffs[..., 0]
        self.vShCoeffs[..., 2] = self.vShCoeffs[..., 0]
        #print(self.vShCoeffs)
        #assert False

        self.vShCoeffs2 = 0.0 * torch.ones([ngamma, self.shBands * self.shBands, 3], dtype=torch.float32, device=self.device)
        self.vShCoeffs2[..., 0, 0] = 0.5
        self.vShCoeffs2[..., 2, 0] = -0.5
        self.vShCoeffs2[..., 1] = self.vShCoeffs2[..., 0]
        self.vShCoeffs2[..., 2] = self.vShCoeffs2[..., 0]

        self.vShCoeffs3 = 0.5 * torch.ones([ngamma, self.shBands * self.shBands, 3], dtype=torch.float32, device=self.device)

        #initialize lighting conditions
        if self.cnum > 1:
            length=self.config.gamma_length#porpics1.0, other1.5
            coefflist=[]
            for i in range(self.cnum):
                val=-length+(2*length/self.cnum)*(i+1)
                coefflist.append(val*torch.ones([ngamma,1, self.shBands * self.shBands, 3], dtype=torch.float32, device=self.device))
            self.vShCoeffsn = torch.cat(coefflist,dim=1)
        else:
            self.vShCoeffsn = self.vShCoeffs.unsqueeze(0).clone().detach()

        #self.vShCoeffsn = self.vShCoeffsn + 0.01*(1-2*torch.rand((self.vShCoeffsn.shape)).to(self.device))

        #self.maskcen = torch.tensor([[self.renderer.screenWidth,self.renderer.screenHeight]]*n).to(self.device)
        self.maskcen = 0.1*torch.ones([n,2],dtype=torch.float32,device=self.device)
        self.maskr = 1*torch.ones([n, 1], dtype=torch.float32, device=self.device)
        self.maskl = torch.zeros([1, self.cnum], dtype = torch.float32, device = self.device)
        self.tao = torch.ones([n,1], dtype = torch.float32, device = self.device)


        texRes = self.morphableModel.getTextureResolution()
        self.vRoughness = 0.4 * torch.ones([nShape, texRes, texRes, 1], dtype=torch.float32, device=self.device)

        xcoor=torch.tensor([list(range(self.renderer.screenWidth))]*self.renderer.screenHeight,dtype=float).to(self.device)
        ycoor=torch.tensor([list(range(self.renderer.screenHeight))]*self.renderer.screenWidth,dtype=float).to(self.device)
        ycoor=ycoor.transpose(0,1)
        coors=torch.cat([xcoor.unsqueeze(-1),ycoor.unsqueeze(-1)],dim=-1)
        self.coors=coors.unsqueeze(0).repeat((n,1,1,1))/(self.renderer.screenHeight*1.0)

        tcoor=torch.tensor([list(range(n))],dtype=float).to(self.device).squeeze(0)
        tcoor=tcoor.unsqueeze(-1).unsqueeze(1).unsqueeze(2)
        #print(tcoor.shape)
        tcoor=tcoor.repeat((1,self.renderer.screenHeight,self.renderer.screenWidth,1))/(n*1.0) * (self.renderer.screenWidth/(self.renderer.screenWidth-1))
        self.acoors=torch.cat([self.coors,tcoor],dim=-1)

        self.softmax=nn.Softmax(dim=1)
        self.softmax1=nn.Softmax(dim=0)

        self.folding = nn.Sequential(
            nn.Conv2d(3, 16, 1, stride=1, padding=0),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(16, 64, 1, stride=1, padding=0),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(64, self.cnum, 1, stride=1, padding=0),
            ).to(self.device)
        self.maskfunc = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=1, padding=2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(16, 64, 1, stride=1, padding=0),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(64, 1, 1, stride=1, padding=0),
            ).to(self.device)
        


    def getsmask(self):
        #self.coors:n*h*w*2
        #self.maskcen:n*2
        #self.maskr:n*1
        maskcen=self.maskcen.unsqueeze(1).unsqueeze(2)
        maskr=self.maskr.unsqueeze(1)#.unsqueeze(2)
        dis=torch.sum(torch.abs((maskcen-self.coors/self.renderer.screenWidth)),dim=-1)

        mask1=torch.exp(-dis/(1e-5+maskr)).unsqueeze(-1)
        mask2=torch.ones_like(mask1)-mask1
        return mask1,mask2

    def getlayers(self):
        coors=self.acoors.permute(0,3,1,2)#/255.0
        coors=2*coors-1
        coors=coors.to(self.device)#.float()
        maski=self.folding(coors.float())
        maskr=self.maskr.unsqueeze(2).unsqueeze(3)
        masks=self.softmax(maski/(1e-5+maskr**2))
        masks=masks.permute(0,2,3,1)
        return masks[:,:,:,:1],masks[:,:,:,1:]

    def getfmask(self):
        coors=self.acoors.permute(0,3,1,2)#/255.0
        coors=2*coors-1
        coors=coors.to(self.device)#.float()
        maski=self.maskfunc(coors.float())
        #maskr=self.tao.unsqueeze(2).unsqueeze(3)
        #masks=self.softmax(maski/(1e-5+maskr**2))
        masks = torch.sigmoid(maski)#-1#+1e-10 exp
        #maski=torch.exp(maski/(1e-5+maskr))
        #masks=torch.exp(maski)/torch.exp()
        masks=masks.permute(0,2,3,1)
        #print(masks.shape)
        #assert False
        return masks#[:,:,:,:1]

    def getfconv(self,imgin,imgout,softdis):
        coors=self.acoors.permute(0,3,1,2)#/255.0
        coors=coors.to(self.device).float()
        #print(coors)
        #assert False
        coors=torch.cat([coors,softdis.permute(0,3,1,2)],axis=1)
        maski=self.maskfunc(coors.float())
        maskr=self.tao.unsqueeze(2).unsqueeze(3)
        #masks=self.softmax(maski/(1e-5+maskr**2))
        masks = torch.exp(maski)#-1#+1e-10 exp
        #maski=torch.exp(maski/(1e-5+maskr))
        #masks=torch.exp(maski)/torch.exp()
        masks=masks.permute(0,2,3,1)
        #print(masks.shape)
        #assert False
        return masks#[:,:,:,:1]

    #Get light masks
    def ftlayers(self):
        maskl=torch.sigmoid(self.maskl.unsqueeze(2).unsqueeze(3))

        coors=self.acoors.permute(0,3,1,2)#/255.0
        coors=coors.to(self.device)#.float()
        coors=2*coors-1
        maski=self.folding(coors.float())
        masks=self.softmax(maski)#/(1e-12+maskr**2))
        return masks,maskl


    def computeShape(self):
        '''
        compute shape vertices from the shape and expression coefficients
        :return: tensor of 3d vertices [n, verticesNumber, 3]
        '''

        assert(self.vShapeCoeff is not None and self.vExpCoeff is not None)
        vertices = self.morphableModel.computeShape(self.vShapeCoeff, self.vExpCoeff)
        return vertices

    def transwithrot(self, vertices = None, vRotation = None):
        '''
        transform vertices to camera coordinate space
        :param vertices: tensor of 3d vertices [n, verticesNumber, 3]
        :return:  transformed  vertices [n, verticesNumber, 3]
        '''

        if vertices is None:
            vertices = self.computeShape()
        if vRotation is None:
            vRotation = self.vRotation

        assert(vertices.dim() == 3 and vertices.shape[-1] == 3)
        assert(self.vTranslation is not None and self.vRotation is not None)
        assert(vertices.shape[0] == self.vTranslation.shape[0] == self.vRotation.shape[0])

        transformedVertices = self.camera.transformVertices(vertices, self.vTranslation, vRotation)
        return transformedVertices

    def transformVertices(self, vertices = None):
        '''
        transform vertices to camera coordinate space
        :param vertices: tensor of 3d vertices [n, verticesNumber, 3]
        :return:  transformed  vertices [n, verticesNumber, 3]
        '''

        if vertices is None:
            vertices = self.computeShape()

        assert(vertices.dim() == 3 and vertices.shape[-1] == 3)
        assert(self.vTranslation is not None and self.vRotation is not None)
        assert(vertices.shape[0] == self.vTranslation.shape[0] == self.vRotation.shape[0])

        transformedVertices = self.camera.transformVertices(vertices, self.vTranslation, self.vRotation)
        return transformedVertices

    def render(self, cameraVerts = None, diffuseTextures = None, specularTextures = None, roughnessTextures = None, normals=None,shcoeffs=None, renderAlbedo = False, render512 = False):
        '''
        ray trace an image given camera vertices and corresponding textures
        :param cameraVerts: camera vertices tensor [n, verticesNumber, 3]
        :param diffuseTextures: diffuse textures tensor [n, texRes, texRes, 3]
        :param specularTextures: specular textures tensor [n, texRes, texRes, 3]
        :param roughnessTextures: roughness textures tensor [n, texRes, texRes, 1]
        :param renderAlbedo: if True render albedo else ray trace image
        :return: ray traced images [n, resX, resY, 4]
        '''
        if cameraVerts is None:
            vertices, diffAlbedo, specAlbedo = self.morphableModel.computeShapeAlbedo(self.vShapeCoeff, self.vExpCoeff, self.vAlbedoCoeff)
            cameraVerts = self.camera.transformVertices(vertices, self.vTranslation, self.vRotation)

        #compute normals
        if normals is None:
            normals = self.morphableModel.meshNormals.computeNormals(cameraVerts)

        if diffuseTextures is None:
            diffuseTextures = self.morphableModel.generateTextureFromAlbedo(diffAlbedo)

        if specularTextures is None:
            specularTextures = self.morphableModel.generateTextureFromAlbedo(specAlbedo)

        if roughnessTextures is None: 
            roughnessTextures  = self.vRoughness

        if shcoeffs is None:
            envMaps = self.sh.toEnvMap(self.vShCoeffs)
        else:
            envMaps = self.sh.toEnvMap(shcoeffs)
        envMaps = envMaps.repeat((cameraVerts.shape[0],1,1,1))
        #print(cameraVerts.shape,envMaps.shape)

        assert(envMaps.dim() == 4 and envMaps.shape[-1] == 3)
        assert (cameraVerts.dim() == 3 and cameraVerts.shape[-1] == 3)
        assert (diffuseTextures.dim() == 4 and diffuseTextures.shape[1] == diffuseTextures.shape[2] == self.morphableModel.getTextureResolution() and diffuseTextures.shape[-1] == 3)
        assert (specularTextures.dim() == 4 and specularTextures.shape[1] == specularTextures.shape[2] == self.morphableModel.getTextureResolution() and specularTextures.shape[-1] == 3)
        assert (roughnessTextures.dim() == 4 and roughnessTextures.shape[1] == roughnessTextures.shape[2] == self.morphableModel.getTextureResolution() and roughnessTextures.shape[-1] == 1)
        assert(cameraVerts.shape[0] == envMaps.shape[0])
        assert (diffuseTextures.shape[0] == specularTextures.shape[0] == roughnessTextures.shape[0])
        #print(diffuseTextures,specularTextures)

        #print(self.uvMap)
        cwidth = self.renderer.screenWidth
        chight = self.renderer.screenHeight
        if render512:
            self.renderer.screenWidth=512
            self.renderer.screenHeight=512

        scenes = self.renderer.buildScenes(cameraVerts, self.faces32, normals, self.uvMap, diffuseTextures,
                                           specularTextures, torch.clamp(roughnessTextures, 1e-20, 10.0), self.vFocals, envMaps)
        self.renderer.screenWidth=cwidth
        self.renderer.screenHeight=chight
        #assert False
        #print(scenes)
        if renderAlbedo:
            images = self.renderer.renderAlbedo(scenes)
        else:
            images = self.renderer.render(scenes)

        #if render512:
        #    cwidth = self.renderer.screenWidth
        #    chight = self.renderer.screenHeight
        #    self.renderer.screenWidth=512
        #    self.renderer.screenHeight=512
        #    scenes = self.renderer.buildScenes(cameraVerts, self.faces32, normals, self.uvMap, diffuseTextures,
        #                                   specularTextures, torch.clamp(roughnessTextures, 1e-20, 10.0), self.vFocals, envMaps)
        #    images2 = self.renderer.render(scenes)
        #    self.renderer.screenWidth=cwidth
        #    self.renderer.screenHeight=chight
        #    return images, images2
        ##print(images)
        #images = torch.clamp(images, 0.0, 1.0)
        return images

    def landmarkLoss(self, cameraVertices, landmarks, focals, cameraCenters,  debugDir = None):
        '''
        calculate scalar loss between vertices in camera space and 2d landmarks pixels
        :param cameraVertices: 3d vertices [n, nVertices, 3]
        :param landmarks: 2d corresponding pixels [n, nVertices, 2]
        :param landmarks: camera focals [n]
        :param cameraCenters: camera centers [n, 2
        :param debugDir: if not none save landmarks and vertices to an image file
        :return: scalar loss (float)
        '''
        assert (cameraVertices.dim() == 3 and cameraVertices.shape[-1] == 3)
        assert (focals.dim() == 1)
        assert(cameraCenters.dim() == 2 and cameraCenters.shape[-1] == 2)
        assert (landmarks.dim() == 3 and landmarks.shape[-1] == 2)
        assert cameraVertices.shape[0] == landmarks.shape[0] == focals.shape[0] == cameraCenters.shape[0]

        headPoints = cameraVertices[:, self.morphableModel.landmarksAssociation]
        assert (landmarks.shape[-2] == headPoints.shape[-2])
        #print(headPoints.shape,headPoints[...].shape,headPoints[..., :2].shape)
        #assert False

        projPoints = focals.view(-1, 1, 1) * headPoints[..., :2] / headPoints[..., 2:]
        #print(cameraCenters)
        #assert False
        projPoints += cameraCenters.unsqueeze(1)
        loss = torch.norm(projPoints - landmarks, 2, dim=-1).pow(2).mean()
        if debugDir:
            for i in range(projPoints.shape[0]):
                image = saveLandmarksVerticesProjections(self.inputImage.tensor[i], projPoints[i], self.landmarks[i])
                cv2.imwrite(debugDir + '/lp' +  str(i) +'.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return loss
