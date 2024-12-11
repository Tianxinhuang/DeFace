from optimizer import *
import argparse

def swap_pics(indir, outdir, ckptdir, config):
    idnames = os.listdir(indir)
    idnames = ['id9180']
    for i in range(len(idnames)):
        idi = idnames[i]
        #if idi in ['id06000']:
        #    config.lamdmarksDetectorType = 'fan'
        #else:
        #    config.lamdmarksDetectorType == 'mediapipe'
        idpath = os.path.join(indir, idi)
        pairnames = os.listdir(idpath)
        pairnames = ['pair1']
        for pair in pairnames:
            outpath = os.path.join(outdir, idi, pair)
            ckpt_path = os.path.join(ckptdir, idi, pair)

            if not os.path.exists(outpath):
                os.makedirs(outpath)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)

            pairpath = os.path.join(idpath, pair)

            sourcein_path = os.path.join(pairpath, 'source')
            targetin_path = os.path.join(pairpath, 'target')

            swap_paras(sourcein_path, targetin_path,os.path.join(outdir, idi, pair, 'source'), os.path.join(outdir, idi, pair, 'target'), os.path.join(ckptdir, idi, pair, 'source.pickle'), os.path.join(ckptdir, idi, pair, 'target.pickle'), config)
            assert False

def swap_paras(source, target, soutpath, outpath, source_ckpt, target_ckpt, config):
    op_source = Optimizer(outpath, config)
    op_source.setImage(source, True)
    if not os.path.exists(source_ckpt):
        op_source.runStep1()
        op_source.runStep2()
        op_source.runStep3()
        op_source.saveParameters(source_ckpt)
    else:
        #If exist, directly using former saved ckpts
        op_source.loadParameters(source_ckpt)
    op_source.savepics(config.rtSamples, soutpath)
    lnum = config.lnum

    op_target = Optimizer(outpath, config)
    op_target.setImage(target, True)

    if not os.path.exists(target_ckpt):
        #op_target.setImage(target, True)
        op_target.runStep1()
        op_target.runStep2()
        op_target.runStep3()
        op_target.saveParameters(target_ckpt)
    else:
        op_target.loadParameters(target_ckpt)

    op_target.vEnhancedDiffuse = op_source.vEnhancedDiffuse
    op_target.vEnhancedSpecular = op_source.vEnhancedSpecular
    op_target.vEnhancedRoughness = op_source.vEnhancedRoughness
    op_target.savepics(config.rtSamples, outpath)


def genpics(indir, outdir, ckptdir, config):
    picnames = os.listdir(indir)
    picnames = ['1112.jpg']
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    for pname in picnames:
        inpath = os.path.join(indir, pname)
        if os.path.isfile(inpath): 
            outpath = os.path.join(outdir, pname.split('.')[0])
            ckpt_path = os.path.join(ckptdir, pname.split('.')[0]+'_ckpt.pickle')
            if os.path.exists(ckpt_path):
                continue

            generate(inpath, outpath, ckpt_path, config)
            

def generate(inpath, outpath, ckpt_path, config):
    print(inpath, outpath, ckpt_path)
    if not os.path.exists(ckpt_path):
        op = Optimizer(outpath, config)
        op.setImage(inpath, True)
        op.runStep1()
        op.runStep2()
        op.runStep3()

        op.saveParameters(ckpt_path)
        op.savepics(config.rtSamples, outpath)
    else:
        op = Optimizer(outpath, config)
        op.setImage(inpath, True)

        op.loadParameters(ckpt_path)
        op.savepics(config.rtSamples, outpath)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=False, default='/Light_distangle/Deface/cfgs/vox2_img.ini')
    parser.add_argument("--input_dir", required=False)
    parser.add_argument("--ckpt_dir", required=False)
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
    if 'celeba' in params.input_dir:
        genpics(params.input_dir, params.output_dir, params.ckpt_dir, config)
    else:
        swap_pics(params.input_dir, params.output_dir, params.ckpt_dir, config)
    #genpics('/Light_distangle/Data/occface/celeba512', '/Light_distangle/Data/faceresults/deface/pics/celeba512_new3', '/Light_distangle/Data/deface/ckpts/celeba512_new3', config)
    #swap_pics('/Light_distangle/Data/occface/porpics3_2', '/Light_distangle/Data/faceresults/deface/pics/porpics3', '/Light_distangle/Data/deface/ckpts/porpics3_2', config)
    #swap_pics('/Light_distangle/Data/occface/voxceleb0_new', '/Light_distangle/Data/faceresults/deface/pics/voxceleb0_B3', '/Light_distangle/Data/deface/ckpts/voxceleb0_B3', config)

