# Script to transfer weights from the unconditional model 
# to the conditional model

import pickle

def percent(current, total):
	if current == 0 or total == 0: return 0
	else:
		perc = current/(total/100.0)
		return float("%.2f"%perc)

class WeightTransfer():
	def __init__(self, srcPath, destPath, useSubnets=["G","G_ema","D"], transferMapping=True):
		self.srcPath, self.destPath = srcPath, destPath
		self.useSubnets, self.transferMapping = useSubnets, transferMapping
		self.allSubnets = ["D","G","G_ema"]
		self.pickles = None

	def transfer(self, outPath):
		if self.pickles == None: self.loadPickles()
		mapping = self.findMapping(self.pickles)
		count, mappedCount, saveData = 0, 0, {}
		for subNet in mapping:
			srcState, destState = self.pickles["src"][subNet].state_dict(), self.pickles["dest"][subNet].state_dict()
			for param in mapping[subNet]:
				m = mapping[subNet][param]
				count+=1
				if m != None:
					destState[m] = srcState[param]
					mappedCount+=1
			self.pickles["dest"][subNet].load_state_dict(destState)		
		print("Transferred",mappedCount,"/",count,"parameters (",percent(mappedCount,count),"%)")
		for subNet in self.allSubnets: saveData[subNet] = self.pickles["dest"][subNet]
		with open(outPath, 'wb') as f: pickle.dump(saveData, f)

	def findMapping(self, pickles):
		result = {}
		for subNet in self.useSubnets:
			srcParams, destParams = pickles["src"][subNet+"_params"], pickles["dest"][subNet+"_params"]
			srcState, destState = pickles["src"][subNet].state_dict(), pickles["dest"][subNet].state_dict()
			result[subNet] = {}
			for paramName in srcParams:
				if paramName in destParams and srcState[paramName].shape == destState[paramName].shape: result[subNet][paramName] = paramName
				else: result[subNet][paramName] = None
		return result

	def loadPickles(self):
		if self.pickles == None:
			self.pickles = {}
			for key, pklFile in [("src",self.srcPath),("dest",self.destPath)]:
				self.pickles[key] = {"picklePath":pklFile}
				with open(pklFile,"rb") as f: 
					pkl = pickle.load(f)
					self.pickles[key]["res"] = pkl["G"].__dict__["img_resolution"]
					for subNet in self.allSubnets:
						self.pickles[key][subNet] = pkl[subNet]
						self.pickles[key][subNet+"_params"] = list(name for name,weight in pkl[subNet].named_parameters())

path_to_uncond_model_weights = "models/network-snapshot-009421.pkl"
path_to_cond_model_weights = "models/network-snapshot-000001.pkl"

path_to_transfer_weights = "models/cond-weight-transfer-resume.pkl"

wt = WeightTransfer(path_to_uncond_model_weights, path_to_cond_model_weights)
wt.transfer(path_to_transfer_weights)
