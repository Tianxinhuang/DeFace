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
sys.path.append('/apdcephfs/private_yaoshihuang/CPEM/core/')
sys.path.append('/apdcephfs/private_yaoshihuang/CPEM')
from data_utils import Preprocess,crop_img,parse_roi_box_from_bbox,crop_and_resize_by_bbox
from FaceBoxes import FaceBoxes


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
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

    im = np.array(im)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) #+ 255
    
    num_of_class = np.max(vis_parsing_anno)
    facelist=[1,2,3,10,12,13]
    for pi in range(1, num_of_class + 1):
        if pi not in facelist:
            continue
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = [255,255,255]#part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im=vis_parsing_anno_color
    vis_im=cv2.resize(vis_im, dsize=(224,224), interpolation=cv2.INTER_LINEAR)
    # Save result or not
    if save_im:
        #cv2.imwrite(save_path[:-4] +'.png', im)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #assert False

    # return vis_im

def evaluate(respth='./res/test_res', dspth='./data',lm_path='',lm2d_path='', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('/apdcephfs/private_yaoshihuang/faceparsing/res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img=cv2.imread(osp.join(dspth, image_path))
            
            face_detector = FaceBoxes()
            boxes = face_detector(img)
            #image_path=''
            #print(image_path)
            #assert False
            lmpath=os.path.join(lm_path,image_path.split('.')[0]+'.txt')
            lm2dpath=os.path.join(lm2d_path,image_path.split('.')[0]+'.txt')
            #print(lm_path)
            if not (os.path.exists(lmpath) and os.path.exists(lm2dpath)):
                #print(lmpath,os.path.exists(lmpath),os.path.exists(lm2dpath))
                #assert False
                continue
            if len(boxes) == 0:
                print('No face detected of {}'.format(image_path))
                #if not (os.path.exists(lmpath) and os.path.exists(lm2dpath)):
                #    continue
                landmark = np.loadtxt(lmpath)  # 68x2
                landmark2d = np.loadtxt(lm2dpath)  # 68x2
                img, new_lm, new_lm2d = crop_and_resize_by_bbox(img, landmark, landmark2d, 224) 
                #continue
            else:
                bbox = boxes[0]
                roi_box = parse_roi_box_from_bbox(bbox)
                img=crop_img(img,roi_box)
            #assert False
            image=cv2.resize(img[:, :, ::-1], dsize=(512,512), interpolation=cv2.INTER_LINEAR)
            img = to_tensor(image).float()
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            #print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))







if __name__ == "__main__":
    #evaluate(dspth='/home/zll/data/CelebAMask-HQ/test-img', cp='79999_iter.pth')
    #paths='HELEN_3214022978_1/'
    #trainpath='/root/htxnet/deconv/CPEM/data/300W_LP_Train/'
    #lmpath=os.path.join(trainpath+'landmarks/'+paths,'001')
    #lmpath2=os.path.join(trainpath+'landmarks2d/'+paths,'001')
    #evaluate(dspth='../CPEM/data/300W_LP_Train/data/HELEN_3214022978_1/001/',lm_path=lmpath,lm2d_path=lmpath2, cp='79999_iter.pth')
    #assert False
    import argparse
    parser = argparse.ArgumentParser()
    trainpath='/apdcephfs/share_1490806/shared_info/htx/voxceleb2'
    parser.add_argument('--video_root', type=str, default=trainpath+'/data')
    parser.add_argument('--start', type=int, default=None, help='start index')
    parser.add_argument('--end', type=int, default=None, help='end index')
    args=parser.parse_args()

    namelist=[]
    with open('/apdcephfs/private_yaoshihuang/difflist.txt','r') as f:
        for line in f.readlines():
            namelist.append(line.strip())

    person_ids = namelist#os.listdir(args.video_root)
    person_ids = [person_id for person_id in person_ids if os.path.isdir(os.path.join(args.video_root, person_id))]
    person_ids.sort()
    person_ids=person_ids[args.start:args.end]
    for person_id in person_ids:
        video_ids=os.listdir(os.path.join(args.video_root,person_id))
        for v_i, video_id in enumerate(video_ids):
            video_path=os.path.join(args.video_root, person_id, video_id)
            view_ids=os.listdir(video_path)
            for v2_i,view_id in enumerate(view_ids):
                #person_id,video_id,view_id='id04599','eYTqzF-zIhM','00175'
                curr_video_path = os.path.join(args.video_root, person_id, video_id,view_id)
                curr_save_path = os.path.join(os.path.join(trainpath,'face_mask'), person_id, video_id,view_id)
                if not os.path.exists(curr_save_path):
                    os.makedirs(curr_save_path)
                #person_id,video_id,view_id='id04599','eYTqzF-zIhM','00175'
                #lmpath=os.path.join(os.path.join(trainpath,'landmarks_s{}_e{}'.format(args.start, args.end)), person_id, video_id,view_id)
                lmpath=os.path.join(os.path.join(trainpath,'landmarks'), person_id, video_id,view_id)
                #lmpath2=os.path.join(os.path.join(trainpath,'landmarks2d_s{}_e{}'.format(args.start, args.end)), person_id, video_id,view_id)
                lmpath2=os.path.join(os.path.join(trainpath,'landmarks2d'), person_id, video_id,view_id)
                despath=curr_save_path
        #for name in os.listdir(files):
        #    filepath=os.path.join(files,name)
        #lmpath=os.path.join(trainpath+'landmarks/'+paths,'001')
        #lmpath2=os.path.join(trainpath+'landmarks2d/'+paths,'001')
        
        #despath=os.path.join(trainpath+'face_mask/'+paths,'001')

                evaluate(respth=despath,dspth=curr_video_path,lm_path=lmpath,lm2d_path=lmpath2,cp='79999_iter.pth')


