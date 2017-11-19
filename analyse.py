import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import maxLikelihood as maxL
import iterativeFiltering as IF

PI = np.pi



def getTemp(x, numReadings):
    return 20 + 5*np.sin(2*PI*x/numReadings - PI/2)

def getRealTemp(numReadings):
    tempReal = []
    for i in range(0,numReadings):
        tempReal.append(getTemp(i,numReadings))

    return tempReal

def getSensorVar(numSensors):
    sensorVar = []
    for i in range(0,numSensors):
        if (i < numSensors-1):
            sensorVar.append(np.random.rand()*15) 
            #original code was (1-15), not (0-15)
        else: 
            sensorVar.append(0.01)

    return sensorVar

def getSensorNoise(numSensors,numReadings,var):
    sensorNoise = []
    for i in range(0,numSensors):
        sensorNoise.append(np.random.normal(0,np.sqrt(var[i]),numReadings))

    return sensorNoise

def getUnsophisticatedMean(sensorReadings):
    meanReadings = []
    for i in range(0,len(sensorReadings[0])): # for every reading
        total = 0
        for j in range(0,len(sensorReadings)-1): # don't include sophisticated sensor
            total += sensorReadings[j][i]
        meanReadings.append(total/(len(sensorReadings)-1))

    return meanReadings

def getSensorReadings(numSensors,numReadings,numColluders, colDiff, attackMode):
    realTemp = getRealTemp(numReadings)
    var = getSensorVar(numSensors)
    noise = getSensorNoise(numSensors,numReadings,var)

    #use maxLikelihoodEstimator here?

    sensorReadings = []
    for i in range(0,numSensors):
        temp = []
        for j in range(0,numReadings):
            if (i <= numSensors - numColluders):
                temp.append(realTemp[j] + noise[i][j])
            else:
                temp.append(realTemp[j] + noise[i][j] + colDiff)
        sensorReadings.append(temp)

    if attackMode == 1:
        # if sophisticated attack, set last collab equal to mean of sensors
        # makes IF algorithm converge to wrong point
        # 
        # does adding noise change this?
        meanReadings = getUnsophisticatedMean(sensorReadings)
        sensorReadings[numSensors-1] = meanReadings
    return sensorReadings


def plotSimple(data,filename='plot.png'):
    plt.figure(figsize=(14,12))  # in inches
    for i in range(0,len(data)):
        x = i
        y = data[i]
        plt.scatter(x, y, c='blue')

    plt.title("Temperature readings")
    plt.xlabel("Reading")
    plt.ylabel("Temp (Degrees C)")
    plt.savefig(filename)
    print("plots saved in {0}".format(filename))

def plotMultipleSimple(data, filename='plot_multiple.png'):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(data))))

    print("past colors")
    plt.figure(figsize=(14,12))  # in inches
    for i in range(0,len(data)):
        c = next(colors)
        x = np.linspace(0, 1, len(data[i][0]))
        plt.plot(x,data[i][0],color=c,label=data[i][1])

    plt.title("Temperature readings")
    plt.xlabel("Reading")
    plt.ylabel("Temp (Degrees C)")
    plt.legend(loc='upper right')
    plt.savefig(filename)
    print("plots saved in {0}".format(filename))

if __name__ == "__main__":
    print("Starting Analysis")

    numSensors = 25   #NN
    numReadings = 288 #TT
    attackMode = 0    #sophisticated
    numColluders = 5  #COL
    colDiff = 100     #M   how much higher than real
    plotOn = True
    iterations = 5

    realTemp = getRealTemp(numReadings)

    total_IF_error = 0
    print("Running %d iterations" % iterations)
    for i in range(0,iterations):
        if (iterations > 20) & (i % (iterations/10) == 0) : print("iteration %d" % i)
        readings = getSensorReadings(numSensors,numReadings,numColluders, 
            colDiff, attackMode)
        IF_weights, IF_counter, IF_estimate, IF_error = IF.IF_algo(readings,realTemp)
        total_IF_error += IF_error
    avg_IF_error = total_IF_error/iterations


    resultsTable = []
    resultsTable.append('IF_error: %f' % IF_error)
    resultsTable.append('avg_IF_error: %f' % avg_IF_error)
    for line in resultsTable:
        print(line) 


    # Plotting
    if (plotOn):
        print("Plotting")
        plotData = []
        plotData.append([realTemp, "realTemp"])
        plotData.append([readings[0], "1st sensor"])
        plotData.append([readings[numSensors-2], "collaborator"])
        if attackMode == 1 :
            plotData.append([readings[numSensors-1], "collaborator (sophisticated)"])
        plotData.append([IF_estimate, "IF_estimate"])
        plotMultipleSimple(plotData)
        plt.show(); #comment out when reading figures seperately

    print("Analysis Finished")
    exit()
