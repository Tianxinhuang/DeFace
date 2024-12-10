#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import sys

def crop_and_resize_by_bbox(image, landmark, img_size):
    # get bbox from landmarks
    left = np.min(landmark[:, 0])
    right = np.max(landmark[:, 0])
    top = np.min(landmark[:, 1])
    bottom = np.max(landmark[:, 1])
    bbox = [left, top, right, bottom]

    roi_box = parse_roi_box_from_landmark_box(bbox)
    img_cropped = crop_img(image, roi_box)
    #print(image, img_cropped)
    img_resized = cv2.resize(img_cropped, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
       
    return img_resized, roi_box
def parse_roi_box_from_landmark_box(bbox):
    """calc roi box from landmark"""
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2 * 1.1
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0

    size = int(old_size * 1.25)
    # size = int(old_size * 1.5)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box

def parse_roi_box_from_bbox(bbox):
    """calc roi box from bounding box"""
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.12
    size = int(old_size * 1.25)
    # size = int(old_size * 1.5)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box
def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    #print(roi_box)
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.float32)
    else:
        res = np.zeros((dh, dw), dtype=np.float32)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh
    #print(dsy,dey,dsx,dex,sy,ey,sx,ex)
    #print(res[dsy:dey, dsx:dex])
    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    #print(res[dsy:dey, dsx:dex], img[sy:ey, sx:ex])
    return res
def overlying_image_origin(roi_box, composed_img, rendered_img):
    '''
    overlay the rendered image on the origin input image
    *** Drawback: there are obvious gaps at the edges of the rendered mask
    '''
    h, w = composed_img.shape[:2]
    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    resized_img = cv2.resize(rendered_img, (dw, dh))  # h,w,3
    #resized_mask = cv2.resize(rendered_mask, (dw, dh))  # h,w,3

    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    composed_img[sy:ey, sx:ex] = resized_img[dsy:dey, dsx:dex] #* resized_mask[dsy:dey, dsx:dex] 
    #+ \composed_img[sy:ey, sx:ex] * (1 - resized_mask[dsy:dey, dsx:dex])

    return composed_img
def vis_parsing_maps(im, parsing_anno, stride, rawsize=(256,256), save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) #+ 255

    num_of_class = np.max(vis_parsing_anno)
    facelist=[1,2,3,10,12,13]
    for pi in range(1, num_of_class + 1):
        if pi not in facelist:
            continue
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = [1.0,1.0,1.0]#part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im=vis_parsing_anno_color
    #print(vis_im.shape)
    vis_im=cv2.resize(vis_im, dsize=rawsize, interpolation=cv2.INTER_LINEAR)
    # Save result or not
    if save_im:
        #cv2.imwrite(save_path[:-4] +'.png', im)
        cv2.imwrite(save_path, (vis_im*255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #assert False

#Get the facial mask by croping the images and project it back
def pic2mask2(img,landmark,cp='79999_iter.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('/Light_distangle/faceparsing/res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    cimg,roi=[],[]
    for i in range(img.shape[0]):
        cimgi, roii = crop_and_resize_by_bbox(img[i].cpu().numpy(), landmark[i], 512)
        cimg.append(torch.tensor(np.expand_dims(cimgi,axis=0)).cuda())
        roi.append(np.expand_dims(roii,axis=0))
    cimg=torch.cat(cimg,dim=0)
    roi=np.concatenate(roi,axis=0)
    oimg=torch.zeros_like(img).cpu()

    torch_resize = transforms.Resize([512,512])
    torch_pad = torch.nn.ZeroPad2d(128)
    t2p = transforms.ToPILImage()
    p2t = transforms.ToTensor()
    img = cimg.permute(0,3,1,2)
    with torch.no_grad():
        img0=img

        mean = torch.tensor([0.485, 0.456, 0.406]).to('cuda')
        std = torch.tensor([0.229, 0.224, 0.225]).to('cuda')

        mean = torch.tensor(mean).reshape(1,-1, 1, 1)
        std = torch.tensor(std).reshape(1,-1, 1, 1)
        # normalize img
        img =  (img - mean) / std
        #img = img.cuda()
        out = net(img)[0]
        #print(out.shape)
        out = out.argmax(1,keepdim=True)
        #all categories belonging to the face regions
        facelist=[1,2,3,4,5,10,11,12,13]
        index=out
        num_of_class=int(out.max())
        for pi in range(1, num_of_class + 1):
            if pi not in facelist:
                continue
            index = torch.where((index-pi).abs()<1e-5, -10*torch.ones_like(out),index)
        mask = torch.where((index+10).abs()<1e-5, 1.0*torch.ones_like(out), 1.0*torch.zeros_like(out)).cpu()
        mask = mask.permute(0,2,3,1).repeat(1,1,1,3)
        #print(mask.max())
        #assert False
        masks=[]
        for i in range(mask.shape[0]):
            masks.append(torch.tensor(overlying_image_origin(roi[i], oimg[i].numpy(), mask[i].cpu().numpy())).unsqueeze(0).cuda())
        mask = torch.cat(masks,dim=0)[...,:1]
        #mask = torch.where(mask>=1.0,  torch.ones_like(mask), torch.zeros_like(mask))
        #mask = mask.permute(0,2,3,1).cuda()#[:,128:384,128:384,:]
        print('mask shape,',mask.shape)
    return mask

def pic2mask(img,cp='79999_iter.pth'):
    #print(img.shape)
    #assert False
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('/Light_distangle/faceparsing/res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    torch_resize = transforms.Resize([512,512])
    torch_pad = torch.nn.ZeroPad2d(128)
    t2p = transforms.ToPILImage()
    p2t = transforms.ToTensor()
    resize_back = transforms.Resize([img.shape[1],img.shape[2]])
    img = img.permute(0,3,1,2)
    with torch.no_grad():
        img0=img
        imgs=[]
        img=img.cpu()
        for i in range(img.shape[0]):
            #print(img[i].shape,p2t(torch_resize(t2p(img[i]))).unsqueeze(0).shape)
            #assert False
            imgs.append(p2t(torch_resize(t2p(img[i]))).unsqueeze(0))
            #imgs.append(torch_pad(img[i].unsqueeze(0)))
        img = torch.cat(imgs,dim=0)
        img=img.cuda()
        #print(img.shape)
        img0=img

        mean = torch.tensor([0.485, 0.456, 0.406]).to('cuda')
        std = torch.tensor([0.229, 0.224, 0.225]).to('cuda')

        mean = torch.tensor(mean).reshape(1,-1, 1, 1)
        std = torch.tensor(std).reshape(1,-1, 1, 1)
        # normalize img
        img =  (img - mean) / std
        #img = img.cuda()
        out = net(img)[0]
        #print(out.shape)
        out = out.argmax(1,keepdim=True)
        #assert False
        facelist=[1,2,3,10,12,13]
        index=out
        #print(index.max())
        num_of_class=int(out.max())
        for pi in range(1, num_of_class + 1):
            if pi not in facelist:
                continue
            index = torch.where((index-pi).abs()<1e-5, -10*torch.ones_like(out),index)
        #print(index.min())
        mask = torch.where((index+10).abs()<1e-5, 1.0*torch.ones_like(out), 1.0*torch.zeros_like(out)).cpu()
        #print(mask.max())
        #assert False
        masks=[]
        for i in range(mask.shape[0]):
            masks.append(p2t(resize_back(t2p(mask[i].float()))).unsqueeze(0))
        #    masks.append(mask[i,:,128:384,128:384].float().unsqueeze(0))
        mask = torch.cat(masks,dim=0)
        #mask = torch.where(mask>=1.0,  torch.ones_like(mask), torch.zeros_like(mask))
        mask = mask.permute(0,2,3,1).cuda()#[:,128:384,128:384,:]
        print('mask shape,',mask.shape)
        #assert False
        #parsing = out.cpu().numpy().argmax(1)
        #print(np.unique(parsing))

        #npmask=vis_parsing_maps(img, parsing, stride=1, save_im=True,rawsize=(img0.shape[-1],img0.shape[-2]), save_path=osp.join(respth, 'test_seg.jpg'))
        #mask=torch.tensor(npmask)
    return mask#img0.permute(0,2,3,1)#[:,128:384,128:384,:]

def evaluate(respth='./', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('/Light_distangle/faceparsing/res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        #assert False
        img=cv2.imread(dspth)
        #image=np.ascontiguousarray(img[:,:,::-1])
        image=cv2.resize(img[:, :, ::-1], dsize=(512,512), interpolation=cv2.INTER_LINEAR)
        #print(image.shape)
        img = to_tensor(image).float()
        #print(img.shape)
        #assert False
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        #print(np.unique(parsing))

        vis_parsing_maps(image, parsing, stride=1, save_im=True,rawsize=(img.shape[-1],img.shape[-2]), save_path=osp.join(respth, 'test_seg.jpg'))







if __name__ == "__main__":
    evaluate(dspth='/Light_distangle/256pics/hatest/21.jpg', cp='79999_iter.pth')
    ##paths='HELEN_3214022978_1/'
    ##trainpath='/root/htxnet/deconv/CPEM/data/300W_LP_Train/'
    ##lmpath=os.path.join(trainpath+'landmarks/'+paths,'001')
    ##lmpath2=os.path.join(trainpath+'landmarks2d/'+paths,'001')
    ##evaluate(dspth='../CPEM/data/300W_LP_Train/data/HELEN_3214022978_1/001/',lm_path=lmpath,lm2d_path=lmpath2, cp='79999_iter.pth')
    ##assert False
    #import argparse
    #parser = argparse.ArgumentParser()
    #trainpath='/apdcephfs/share_1490806/shared_info/htx/voxceleb2'
    #parser.add_argument('--video_root', type=str, default=trainpath+'/data')
    #parser.add_argument('--start', type=int, default=None, help='start index')
    #parser.add_argument('--end', type=int, default=None, help='end index')
    #args=parser.parse_args()

    #namelist=[]
    #with open('/apdcephfs/private_yaoshihuang/difflist.txt','r') as f:
    #    for line in f.readlines():
    #        namelist.append(line.strip())

    #person_ids = namelist#os.listdir(args.video_root)
    #person_ids = [person_id for person_id in person_ids if os.path.isdir(os.path.join(args.video_root, person_id))]
    #person_ids.sort()
    #person_ids=person_ids[args.start:args.end]
    #for person_id in person_ids:
    #    video_ids=os.listdir(os.path.join(args.video_root,person_id))
    #    for v_i, video_id in enumerate(video_ids):
    #        video_path=os.path.join(args.video_root, person_id, video_id)
    #        view_ids=os.listdir(video_path)
    #        for v2_i,view_id in enumerate(view_ids):
    #            #person_id,video_id,view_id='id04599','eYTqzF-zIhM','00175'
    #            curr_video_path = os.path.join(args.video_root, person_id, video_id,view_id)
    #            curr_save_path = os.path.join(os.path.join(trainpath,'face_mask'), person_id, video_id,view_id)
    #            if not os.path.exists(curr_save_path):
    #                os.makedirs(curr_save_path)
    #            #person_id,video_id,view_id='id04599','eYTqzF-zIhM','00175'
    #            #lmpath=os.path.join(os.path.join(trainpath,'landmarks_s{}_e{}'.format(args.start, args.end)), person_id, video_id,view_id)
    #            lmpath=os.path.join(os.path.join(trainpath,'landmarks'), person_id, video_id,view_id)
    #            #lmpath2=os.path.join(os.path.join(trainpath,'landmarks2d_s{}_e{}'.format(args.start, args.end)), person_id, video_id,view_id)
    #            lmpath2=os.path.join(os.path.join(trainpath,'landmarks2d'), person_id, video_id,view_id)
    #            despath=curr_save_path
    #    #for name in os.listdir(files):
    #    #    filepath=os.path.join(files,name)
    #    #lmpath=os.path.join(trainpath+'landmarks/'+paths,'001')
    #    #lmpath2=os.path.join(trainpath+'landmarks2d/'+paths,'001')
    #    
    #    #despath=os.path.join(trainpath+'face_mask/'+paths,'001')

    #            evaluate(respth=despath,dspth=curr_video_path,lm_path=lmpath,lm2d_path=lmpath2,cp='79999_iter.pth')


