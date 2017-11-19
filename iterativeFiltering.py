# Algorithms to estimate the true value from sensor data
#  without knowing variance

import numpy as np
import maxLikelihood as maxL

# realValues only passed in to keep analytics together
def IF_algo(readings,realValues):
	# Initial estimate
	estimate = initialEstimate(readings)

	counter = 0 # counter for number of iterations
	diff = 1000 # starting value for RMS diff between two consecutive iterations
	accuracy = 10**(-8) #threshold when to stop iterating

	while (diff > accuracy):
		oldEstimate = estimate
		# estimation of variances
		var = IF_getVar(readings,estimate)
		#print(counter, var[0], diff)
		weights = IF_getWeights(var)
		estimate = IF_getEstimate(weights, readings)
		diff = maxL.RMSE(estimate, oldEstimate)
		counter += 1

	error = maxL.RMSE(realValues,estimate)	
	return weights, counter, estimate, error

def initialEstimate(readings):
	estimate = []
	for i in range(0,len(readings[0])):
		total = 0
		for j in range(0,len(readings)):
			total += readings[j][i]
		estimate.append(total)

	return estimate

# estimation of variances
def IF_getVar(readings,estimate):
	var = []
	for i in range(0,len(readings)):
		total = 0
		for j in range(0,len(readings[0])):
			total += (readings[i][j] - estimate[j])**2
		var.append(total/(len(readings[0])-1))

	return var

def IF_getWeights(var):
	weights = []
	for i in range(0,len(var)):
		den = 0
		for j in range(0,len(var)):
			den += 1/var[j]
	
		weights.append((1/var[i])/den)

	return weights

def IF_getEstimate(weights, readings):
	estimate = []
	for i in range(0,len(readings[0])):
		total = 0
		for j in range(0,len(readings)):
			total += weights[j]*readings[j][i]
		estimate.append(total)

	return estimate

def IF_Affine_algo(readings,realValues):
	# Initial estimate
	estimate = initialEstimate(readings)

	counter = 0 # counter for number of iterations
	diff = 1000 # starting value for RMS diff between two consecutive iterations
	accuracy = 10**(-8) #threshold when to stop iterating

	while (diff > accuracy):
		oldEstimate = estimate

		# distance of estimate from readings
		dist = IF_Affine_getDist(readings, estimate)



		# estimation of variances
		var = IF_getVar(readings,estimate)
		weights = IF_getWeights(var)
		estimate = IF_getEstimate(weights, readings)
		diff = maxL.RMSE(estimate, oldEstimate)
		counter += 1

	error = maxL.RMSE(realValues,estimate)	
	return weights, counter, estimate, error

def IF_Affine_getDist(readings,estimate):
	dist = []
	for i in range(0,len(readings)): #for each sensor
		squareTotal = 0
		for t in range(0,len(readings[0])):
			squareTotal += (readings[i][t] - estimate[t])**2
		dist.append(np.sqrt(squareTotal))

	return dist