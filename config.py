import copy
import sys


class Config:
	def __init__(self):
		#compute device
            self.device = 'cuda'
            
            #tracker
            self.lamdmarksDetectorType = 'mediapipe'  # Options ['mediapipe', 'fan']
            #morphable model
            self.path = '/Light_distangle/NextFace3/baselMorphableModel'
            self.stylepath = '/Light_distangle/stylegan3/pkls/stylegan3-t-ffhqu-256x256.pkl'
            self.textureResolution = 256 #256 or 512
            self.trimPca = False  # if True keep only a subset of the pca basis (eigen vectors)
            
            #spherical harmonics
            self.bands = 3
            self.envMapRes = 64
            self.smoothSh = False
            self.saveExr = True
            #camera
            self.camFocalLength = 500.0 #focal length in pixels (f =  f_{mm} * imageWidth / sensorWidth)
            self.optimizeFocalLength = True #if True the initial focal length is estimated otherwise it remains constant
            
            #image
            self.maxResolution = 512
            
            #Settings for light decoupling
            self.epsilon = 0.15
            self.w0 = 1000.0
            self.w1 = 1.0
            self.w2 = 150.0
            self.w3 = 0.5
            self.w4 = 20
            self.w5 = 2000.0
            self.w6 = 2.0
            self.w7 = 1.0
            self.gamma_length = 1.5
            self.gamma_lr = 0.01
            self.iterStep0 = 100
            self.lnum = 1
            self.shareid = False

            
            #optimization
            self.iterStep1 = 2000 # number of iterations for the coarse optim
            self.iterStep2 = 100 #number of iteration for the first dense optim (based on statistical priors)
            self.iterStep3 = 100 #number of iterations for refining the statistical albedo priors
            self.weightLandmarksLossStep2 = 0.001 #landmarks weight during step2
            self.weightLandmarksLossStep3 = 0.001  # landmarks weight during step3
            
            self.weightShapeReg = 0.001 #weight for shape regularization
            self.weightExpressionReg = 0.001  # weight for expression regularization
            self.weightAlbedoReg = 0.001  # weight for albedo regularization
            
            self.weightDiffuseSymmetryReg = 50. #symmetry regularizer weight for diffuse texture (at step 3)
            self.weightDiffuseConsistencyReg = 100.  # consistency regularizer weight for diffuse texture (at step 3)
            self.weightDiffuseSmoothnessReg = 0.001  # smoothness regularizer weight for diffuse texture (at step 3)
            
            self.weightSpecularSymmetryReg = 30.  # symmetry regularizer weight for specular texture (at step 3)
            self.weightSpecularConsistencyReg = 2.  # consistency regularizer weight for specular texture (at step 3)
            self.weightSpecularSmoothnessReg = 0.001  # smoothness regularizer weight for specular texture (at step 3)
            
            self.weightRoughnessSymmetryReg = 10.  # symmetry regularizer weight for roughness texture (at step 3)
            self.weightRoughnessConsistencyReg = 0.  # consistency regularizer weight for roughness texture (at step 3)
            self.weightRoughnessSmoothnessReg = 0.002  # smoothness regularizer weight for roughness texture (at step 3)
            
            self.debugFrequency = 10 #display frequency during optimization
            self.saveIntermediateStage = False #if True the output of stage 1 and 2 are saved. stage 3 is always saved which is the output of the optim
            self.verbose = False #display loss on terminal if true
            
            self.rtSamples = 500 #the number of ray tracer samples to render the final output
            self.rtTrainingSamples = 8  # number of ray tracing to use during training
	def fillFromDicFile(self, filePath):
		'''
		overwrite default config
		:param filePath: path to the new config file
		:return:
		'''

		print('loading optim config from: ', filePath)
		fp = open(filePath, 'r')
		assert(fp is not None)
		Lines = fp.readlines()
		fp.close()

		dic = {}

		for line in Lines:
			oLine = copy.copy(line)

			if line[0] == '#' or line[0] == '\n':
				continue
			if '#' in line:
				line = line[0:line.find('#')].strip().replace('\t', '').replace('\n', '')

			if len(line) < 1:
				continue

			keyval = line.split('=')
			if len(keyval) == 2:
				#assert (len(keyval) == 2)
				key = keyval[0].strip()
				val = keyval[1].strip()
				val = val.replace('"', '').replace("'", "").strip()
				dic[key] = val
			else:
				print('[warning] unknown key/val: ', oLine, file=sys.stderr, flush=True)

		for k, v in dic.items():
			aType = type(getattr(self, k)).__name__
			if aType == 'str':
				setattr(self, k, v)
			elif aType == 'bool':
				setattr(self, k, v.lower() == 'true')
			elif aType == 'int':
				setattr(self, k, int(v))
			elif aType == 'float':
				setattr(self, k, float(v))
			else:
				raise RuntimeError("unknown dictionary type: "+ key + "=>" + val)
	def print(self):
		dic = self.__dict__
		for key, val in dic.items():
			print(key, '=>', val)
