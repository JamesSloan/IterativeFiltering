import numpy as np

# requires knowing the sensor variances (IF doesn't know the var)
def getWeights(numSensors, sensorVar, numColluders):
	weights = []
	total = 0
	for i in range(0,numSensors):
		if i < (numSensors-numColluders-1):
			den = 0
			for j in range(0,numSensors-numColluders-1):
				den += 1/var[j]
			weights.append((1/var[i])/den)
			total += (1/var[i])/den
		else: 
			weights.append(0)
	for i in range(0,len(weights)):
		weights[i] = weights[i]/total

	return weights

def getEstimates(weights, readings):
	numReadings = len(readings[0])
	estimates = []
	for t in range(0,numReadings):
		total = 0
		for i in range(0,numSensors):
			total += weights[i]*readings[i][t]
		estimates.append(total)

	return estimates

def RMSE(estimates, realValues):
	assert(len(estimates) == len(realValues))
	squaredError = 0
	for i in range(0,len(estimates)):
		squaredError += (estimates[i] - realValues[i])**2
	RMSE = np.sqrt(squaredError/len(estimates))

	return RMSE

def errorBest(noise, excluding=0): #excluding = 5
	errorBest = 9999
	for i in range(0,len(noise)):
		square = 0
		for t in range(0,len(noise[0])):
			square += noise[t]**2
		RMS = np.sqrt(square/len(noise))
		if RMS < errorBest : errorBest = RMS

	return errorBest

