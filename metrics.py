from optimizer import *
from sewar.full_ref import mse, rmse, psnr, rmse_sw, uqi, ssim, ergas, scc, rase, sam, msssim, vifp, psnrb
from PIL import Image
import numpy as np
import os
import pickle5 as pickle
import lpips
import torch
from torchvision import transforms
import scipy
import argparse

lpips_model = lpips.LPIPS(net="alex")
sys.path.append('faceparsing')

import pyredner

def seg_mask(op, imgpath, config):
    op.setImage(imgpath, True)
    fmask = pic2mask2(torch.pow(op.inputImage.tensor,2.2), op.landmarks.cpu().numpy()).detach().clone()
    return fmask.cpu().numpy()

def savepics(fmask, path):
    pyredner.imwrite(fmask, path, gamma = 1.0)

def eva_pics(op, indir, outdir, resdir, config):
    idnames = os.listdir(indir)
    inresult = []
    outresult= []
    myout = outdir.split('/')[-2]
    form = '.png'
    inter = ''
    perfix = 'face'

    for idi in idnames:
        idpath = os.path.join(indir, idi)
        pairnames = os.listdir(idpath)
        idin=[]
        idout=[]
        for pair in pairnames:
            pairpath = os.path.join(idpath, pair)

            sourcein_path = os.path.join(pairpath, 'source')
            targetin_path = os.path.join(pairpath, 'target')

            source_res = os.path.join(outdir, idi, pair, 'source')
            target_res = os.path.join(outdir, idi, pair, 'target')


            picnames = [name for name in os.listdir(sourcein_path) if os.path.isfile(os.path.join(sourcein_path, name))]
            tpicnames =[name for name in os.listdir(targetin_path) if os.path.isfile(os.path.join(targetin_path, name))]
            #picnames = [name for name in list(sorted(picnames, key=lambda x:int(x.split('.')[0])))]
            #tpicnames = [name for name in list(sorted(tpicnames, key=lambda x:int(x.split('.')[0])))]

            for i in range(len(picnames)):
                pname = picnames[i]
                if not os.path.isfile(os.path.join(sourcein_path, pname)):
                    continue

                inpath = os.path.join(sourcein_path, pname)
                inpics = np.asarray(Image.open(inpath))#.astype(float)

                outpath = os.path.join(source_res, inter, pname.split('.')[0]+'_'+perfix+form)
                mlamda = seg_mask(op, inpath, config)[0]
                if not os.path.isfile(outpath):
                    continue
                outpics = np.asarray(Image.open(outpath))[...,:3].astype(float)

                source_vals = eva(inpics, (mlamda*outpics+(1-mlamda)*inpics.astype(float)).astype(int))
                inresult.append(np.expand_dims(np.array(source_vals),axis=0))

                pname = tpicnames[i]

                inpath = os.path.join(targetin_path, pname)
                if not os.path.isfile(inpath):
                    continue
                inpics = np.asarray(Image.open(inpath))#.astype(float)

                outpath = os.path.join(target_res, inter,  pname.split('.')[0]+'_'+perfix+form)
                if not os.path.isfile(outpath):
                    continue
                outpics = np.asarray(Image.open(outpath))[...,:3].astype(float)
                mlamda = seg_mask(op, inpath, config)[0]

                maskpath = os.path.join(outdir, idi, pair, 'target', 'full_mask_'+str(i)+'.png')

                source2_vals = eva(inpics, (mlamda*outpics+(1-mlamda)*inpics.astype(float)).astype(int))
                outresult.append(np.expand_dims(np.array(source2_vals),axis=0))

    inresult = np.concatenate(inresult,0)
    outresult = np.concatenate(outresult,0)
    
    inresult = np.ma.masked_invalid(inresult)
    outresult = np.ma.masked_invalid(outresult)

    inmean, outmean = inresult.mean(0), outresult.mean(0)

    metrics_list = ['MSE:', 'PSNR:', 'SSIM:', 'LPIPS:']
    f = open(resdir, 'w')
    for i in range(len(metrics_list)):
        f.write(metrics_list[i] + ' source: '+str(inmean[i]) + ' target: '+ str(outmean[i])+'\n')
    f.close()


def eva(inpics, outpics):
    res=[mse(inpics, outpics), psnr(inpics, outpics), ssim(inpics, outpics)[0]]
    image1_tensor = torch.tensor(np.array(inpics)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.tensor(np.array(outpics)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    distance = lpips_model(image1_tensor, image2_tensor)
    res.append(distance.item())
    return np.array(res)


def tensor2img(tensor_img):
    tensor = tensor_img
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0)
    image = tensor.numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=False, default='./cfgs/vox2_img.ini')
    parser.add_argument("--input_dir", required=False)
    parser.add_argument("--result_dir", required=False)
    parser.add_argument("--output_dir", required=False)
    params = parser.parse_args()

    configFile = params.configs
    config = Config()
    config.fillFromDicFile(configFile)
    
    if config.device == 'cuda' and torch.cuda.is_available() == False:
        print('[WARN] no cuda enabled device found. switching to cpu... ')
        config.device = 'cpu'
    
    #check if mediapipe is available
    
    if config.lamdmarksDetectorType == 'mediapipe':
        try:
            from  landmarksmediapipe import LandmarksDetectorMediapipe
        except:
            print('[WARN] Mediapipe for landmarks detection not availble. falling back to FAN landmarks detector. You may want to try Mediapipe because it is much accurate than FAN (pip install mediapipe)')
            config.lamdmarksDetectorType = 'fan'
    
    op = Optimizer('', config)
    eva_pics(op, params.input_dir, params.output_dir, params.result_dir, config)
