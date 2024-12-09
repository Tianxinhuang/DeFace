import numpy as np
import torch
import cv2
import os
import torchvision.models as tmodels
import torch.nn as nn
import torch.nn.functional as F
import copy

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        print('content shape:,',self.target.shape)
        #print(self.target.shape)
        #assert False
        #a,b,c,d = input.size()
        #ifeat = input.reshape(a,b,c*d).unsqueeze(-1)
        #tfeat = self.target.reshape(a,b,1,c*d).unsqueeze(-2)
        #ofeat = (ifeat[:,:]-tfeat[:,:]).abs().mean(1)
        #print(ofeat.shape)
        #self.loss = ofeat.min(-1)[0].mean()

        #print(self.target.shape)
        #self.loss = (input-self.target).abs().min(1)[0].mean()
        return input
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        #self.loss = F.mse_loss(G, self.target)
        #print(G.shape)
        #assert False
        self.loss = (G-self.target).abs().mean()
        return input
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(1,-1, 1, 1)
        self.std = torch.tensor(std).view(1,-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # 特征映射 b=number
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # 我们通过除以每个特征映射中的元素数来“标准化”gram矩阵的值.
    return G.div(a * b * c * d)
content_layers_default = ['conv_4']
#style_layers_default = ['conv_15']
style_layers_default = ['conv_3', 'conv_5', 'conv_9']
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to('cuda')

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            #print(style_img.shape)
            #assert False
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses
def saveObj(filename, materialName, vertices, faces, normals = None, tcoords = None, textureFileName = 'texture.png'):
    '''
    write mesh to an obj file
    :param filename: path to where to save the obj file
    :param materialFileName: material name
    :param vertices:  float tensor [n, 3]
    :param faces: tensor [#triangles, 3]
    :param normals: float tensor [n, 3]
    :param tcoords: float tensor [n, 2]
    :param textureFileName: name of the texture to use with material
    :return:
    '''
    assert(vertices.dim() == 2 and  vertices.shape[-1] == 3)
    assert (faces.dim() == 2 and faces.shape[-1] == 3)

    if normals is not None:
        assert (normals.dim() == 2 and normals.shape[-1] == 3)

    if tcoords is not None:
        assert (tcoords.dim() == 2 and tcoords.shape[-1] == 2)

    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()
    if torch.is_tensor(normals):
        normals = normals.detach().cpu().numpy()
    if torch.is_tensor(tcoords):
        tcoords = tcoords.detach().cpu().numpy()

    assert(isinstance(vertices, np.ndarray))
    assert (isinstance(faces, np.ndarray))
    assert (isinstance(normals, np.ndarray))
    assert (isinstance(tcoords, np.ndarray))

    #write material
    f = open(os.path.dirname(filename) + '/' + materialName, 'w')
    f.write('newmtl material0\n')
    f.write('map_Kd ' + textureFileName + '\n')
    f.close()

    f = open(filename, 'w')
    f.write('###########################################################\n')
    f.write('# OBJ file generated by faceYard 2021\n')
    f.write('#\n')
    f.write('# Num Vertices: %d\n' % (vertices.shape[0]))
    f.write('# Num Triangles: %d\n' % (faces.shape[0]))
    f.write('#\n')
    f.write('###########################################################\n')
    f.write('\n')
    f.write('mtllib ' + materialName + '\n')

    #write vertices
    for v in vertices:
        f.write('v %f %f %f\n' % (v[0], v[1], v[2]))

    # write the tcoords
    if tcoords is not None and tcoords.shape[0] > 0:
        for uv in tcoords:
            f.write('vt %f %f\n' % (uv[0], uv[1]))

    #write the normals
    if normals is not None and normals.shape[0] > 0:
        for n in normals:
            f.write('vn %f %f %f\n' % (n[0], n[1], n[2]))

    f.write('usemtl material0\n')
    #write face indices list
    for t in faces:
        f.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (t[0] + 1, t[0] + 1,t[0] + 1,
                                              t[1] + 1, t[1] + 1,t[1] + 1,
                                              t[2] + 1, t[2] + 1, t[2] + 1))
    f.close()
def saveLandmarksVerticesProjections(imageTensor, projPoints, landmarks):
    '''
    for debug, render the projected vertices and landmakrs on image
    :param images: [w, h, 3]
    :param projPoints: [n, 3]
    :param landmarks: [n, 2]
    :return: tensor [w, h, 3
    '''
    assert(imageTensor.dim() == 3 and imageTensor.shape[-1] == 3 )
    assert(projPoints.dim() == 2 and projPoints.shape[-1] == 2)
    assert(projPoints.shape == landmarks.shape)
    image = imageTensor.clone().detach().cpu().numpy() * 255.
    landmarkCount = landmarks.shape[0]
    for i in range(landmarkCount):
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
        x = projPoints[i, 0]
        y = projPoints[i, 1]
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

    return image

def mkdir_p(path):
    import errno
    import os

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def loadDictionaryFromPickle(picklePath):
    import pickle5 as pickle
    handle = open(picklePath, 'rb')
    assert handle is not None
    dic = pickle.load(handle)
    handle.close()
    return dic
def writeDictionaryToPickle(dict, picklePath):
    import pickle5 as pickle
    handle = open(picklePath, 'wb')
    pickle.dump(dict, handle, pickle.HIGHEST_PROTOCOL)
    handle.close()
