import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import tabulate

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

def getSensorVar(numSensors, attackMode):
    sensorVar = []
    for i in range(0,numSensors):
        if attackMode == 0:
            sensorVar.append(np.random.rand()*15)
        else:
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
    for t in range(0,len(sensorReadings[0])): # for every reading
        total = 0
        for i in range(0,len(sensorReadings)-1): # don't include sophisticated sensor
            total += sensorReadings[i][t]
        meanReadings.append(total/(len(sensorReadings)-1))

    return meanReadings

def getSensorReadings(numSensors,numReadings,numColluders, colDiff, attackMode):
    realTemp = getRealTemp(numReadings)
    var = getSensorVar(numSensors,attackMode)
    noise = getSensorNoise(numSensors,numReadings,var)

    #use maxLikelihoodEstimator here?

    sensorReadings = []
    for i in range(0,numSensors):
        temp = []
        for t in range(0,numReadings):
            if (i <= numSensors - numColluders):
                temp.append(realTemp[t] + noise[i][t])
            else:
                temp.append(realTemp[t] + noise[i][t] + colDiff)
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

def tableEntryFormat(name,array):
    temp = []
    temp.append(name)
    for i in range(0,len(array)):
        temp.append(IF_error[i])

    return temp

def runAlgorithms(numSensors,numReadings,numColluders,colDiff,attackMode):
    global iterations
    realTemp = getRealTemp(numReadings)
    total_IF_error = 0
    total_IF_A_error = 0
    for i in range(0,iterations):
        if (iterations > 20) & (i % (iterations/10) == 0) : print("iteration %d" % i)
        readings = getSensorReadings(numSensors,numReadings,numColluders, 
            colDiff, attackMode)

        start = time.process_time()
        IF_weights, IF_counter, IF_estimate, IF_error = IF.IF_algo(readings,realTemp)
        end = time.process_time()
        IF_times.append(end - start)

        start = time.process_time()
        IF_A_weights, IF_A_counter, IF_A_estimate, IF_A_error = IF.IF_Affine_algo(readings,realTemp)
        end = time.process_time()
        IF_A_times.append(end - start)

        total_IF_error += IF_error
        total_IF_A_error += IF_A_error

    avg_IF_error = total_IF_error/iterations
    avg_IF_A_error = total_IF_A_error/iterations

    avg_IF_time = np.mean(IF_times)
    avg_IF_A_time = np.mean(IF_A_times)

    results = []
    results.append([avg_IF_error, avg_IF_time, IF_estimate])
    results.append([avg_IF_A_error, avg_IF_A_time, IF_A_estimate])

    return results

if __name__ == "__main__":
    print("Starting Analysis")

    sensorOptions = [10,20,25,50,75,100] #,200,400,800]
    colPercentOptions = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    colDiffOptions = [5,10,20,40,80,100,200,400,800]
    readingsOptions = [5,10,20,40,80,100,200,400,800]
    attackModeOptions = [0,1]

    default_numSensors = 25                #NN 
    default_numReadings = 288              #TT
    default_numColluders = 5               #COL 
    default_colDiff = 100                  #M:  how much higher than real
    default_attackMode = 1                 #sophisticated
    
    plotOn = False
    global iterations
    iterations = 1


    total_IF_error = 0
    total_IF_A_error = 0
    IF_times = []
    IF_A_times = []
    print("Running %d iterations\n" % iterations)

    # Varying numSensors
    varySensorResults = []
    for attackMode in attackModeOptions:
        temp = []
        for s in sensorOptions:
            print(s)
            numSensors = s
            numReadings = default_numReadings
            numColluders = default_numColluders
            colDiff = default_colDiff
            temp.append(runAlgorithms(
                numSensors,numReadings,numColluders,colDiff,attackMode))
        varySensorResults.append(temp)

    #avg_soph = np.sum(readings[numSensors-1])/numReadings
    #print("SophMean: %f" % (avg_soph/numSensors))

    print(len(varySensorResults))
    print(len(varySensorResults[0]))
    print(len(varySensorResults[0][0]))
    print(len(varySensorResults[0][0][0]))

    # Results table
    resultsTable_unSo = []
    resultsTable_so = []
    IF_error = [] 
    IF_A_error = []
    IF_time = []
    IF_A_time = []
    for a in attackModeOptions:
        temp_IF_error = [] 
        temp_IF_A_error = []
        temp_IF_time = []
        temp_IF_A_time = []
        for s in range(0,len(sensorOptions)):
            temp_IF_error.append(varySensorResults[a][s][0][0])
            temp_IF_A_error.append(varySensorResults[a][s][1][0])
            temp_IF_time.append(varySensorResults[a][s][0][1])
            temp_IF_A_time.append(varySensorResults[a][s][1][1])
        IF_error.append(temp_IF_error) 
        IF_A_error.append(temp_IF_A_error)
        IF_time.append(temp_IF_time)
        IF_A_time.append(temp_IF_A_time)

    unSo_IF_error = tableEntryFormat('IF_error:',F_error[0])
    """
    temp = []
    temp.append('IF_error:')
    for i in range(0,len(IF_error[0])):
        temp.append(IF_error[0][i])
    unSo_IF_error = temp
    """

    temp = []
    temp.append('IF_error:')
    for i in range(0,len(IF_error[1])):
        temp.append(IF_error[1][i])
    so_IF_error = temp

    temp = []
    temp.append('IF_A_error:')
    for i in range(0,len(IF_A_error[0])):
        temp.append(IF_A_error[0][i])
    unSo_IF_A_error = temp

    temp = []
    temp.append('IF_A_error:')
    for i in range(0,len(IF_A_error[1])):
        temp.append(IF_A_error[1][i])
    so_IF_A_error = temp

    temp = []
    temp.append('avg_IF_time:')
    for i in range(0,len(IF_time[0])):
        temp.append(IF_time[0][i])
    unSo_IF_time = temp

    temp = []
    temp.append('avg_IF_time:')
    for i in range(0,len(IF_time[1])):
        temp.append(IF_time[1][i])
    so_IF_time = temp

    temp = []
    temp.append('avg_IF_A_time:')
    for i in range(0,len(IF_A_time[0])):
        temp.append(IF_A_time[0][i])
    unSo_IF_A_time = temp

    temp = []
    temp.append('avg_IF_A_time:')
    for i in range(0,len(IF_A_time[1])):
        temp.append(IF_A_time[1][i])
    so_IF_A_time = temp

    
    # unsophisticated attack
    resultsTable_unSo.append(unSo_IF_error)
    resultsTable_unSo.append(unSo_IF_A_error)  
    resultsTable_unSo.append(unSo_IF_time)
    resultsTable_unSo.append(unSo_IF_A_time)
    # sophisticated attack
    resultsTable_so.append(so_IF_error)
    resultsTable_so.append(so_IF_A_error)  
    resultsTable_so.append(so_IF_time)
    resultsTable_so.append(so_IF_A_time)

    table_Unso = tabulate.tabulate(resultsTable_unSo, 
        sensorOptions, tablefmt="simple", floatfmt=".4f")
    table_so = tabulate.tabulate(resultsTable_so, 
        sensorOptions, tablefmt="simple", floatfmt=".4f")  
    print("Unsophisticated attack")
    print(table_Unso)
    print("Sophisticated attack")
    print(table_so)
    # write table to file
    f = open('results_table.txt', 'w')
    f.write("Varying number of Sensors")
    f.write("\n\nUnsophisticated attack\n")
    f.write(table_Unso)
    f.write("\n\nSophisticated attack\n")
    f.write(table_so)
    f.close()

    
    realTemp = getRealTemp(default_numReadings)

    # Plotting
    if (plotOn):
        print("\nPlotting")
        plotData = []
        plotData.append([realTemp, "realTemp"])
        plotData.append([readings[0], "1st sensor"])
        plotData.append([readings[numSensors-2], "collaborator"])
        if attackMode == 1 :
            plotData.append([readings[numSensors-1], "collaborator (sophisticated)"])
        plotData.append([IF_estimate, "IF_estimate"])
        plotData.append([IF_A_estimate, "IF_Affine_estimate"])
        plotMultipleSimple(plotData)
        plt.show(); #comment out when reading figures seperately

    print("Analysis Finished")
    exit()