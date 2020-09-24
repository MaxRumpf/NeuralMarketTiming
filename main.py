class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

SHORT = False

def dateToYYYYMMDD(date, *args):
    day = date.day
    if len(args) > 0:
        day = args[0]
    return str(date.year) + "-" + str(date.month) + "-" + str(day)


import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import quandl

quandl.ApiConfig.api_key = "ku1U8pjGaWCy9sJMiFWG"

def getDataset(symbol, duration):
    stock = yf.Ticker(symbol)
    dataset = stock.history(period=str(duration) + "y")

    closeData = dataset.loc[:, "Close"].to_frame()
    closeData["Close"] = (closeData["Close"] / closeData.loc[:, "Close"][0]) * 100
    for ma_Duration in [50, 100, 200]:
        movingAverageDatapoints = []
        cumulatedResult = []
        for date, row in closeData.iterrows():
            #print(cumulatedResult)
            if len(cumulatedResult) > ma_Duration:
                cumulatedResult.pop(0)
            cumulatedResult.append(row["Close"])
            calc_Result = 0
            for dataPoint in cumulatedResult:
                calc_Result = calc_Result + dataPoint
            calc_Result = calc_Result / len(cumulatedResult)
            movingAverageDatapoints.append(calc_Result)
        closeData["sma" + str(ma_Duration)] = movingAverageDatapoints

    _dateArray = []
    _monthlyReturns = []
    for date, row in closeData.iterrows():
        _dateArray.append(date)
        if len(_monthlyReturns) == 0:
            _monthlyReturns.append(0)
            continue
        initialPrice = closeData.loc[:, "Close"][_dateArray[0]]
        currentPrice = row["Close"]
        priceDifference = currentPrice - initialPrice
        if priceDifference > 0:
            _monthlyReturns.append(1)
        else:
            _monthlyReturns.append(0)
        if len(_dateArray) > 30:
            _dateArray.pop(0)
        #print(_monthlyReturns)
    closeData["outcome"] = _monthlyReturns


    closeData["50-100"] = closeData["sma50"] - closeData["sma100"]
    closeData["50-200"] = closeData["sma50"] - closeData["sma200"]
    closeData["100-200"] = closeData["sma100"] - closeData["sma200"]
    closeData["spotTo50"] = closeData["Close"] - closeData["sma50"]
    closeData["spotTo100"] = closeData["Close"] - closeData["sma100"]
    closeData["spotTo200"] = closeData["Close"] - closeData["sma200"]

    peRatioData = quandl.get("MULTPL/SP500_PE_RATIO_MONTH")

    vixData = [quandl.get("CHRIS/CBOE_VX1"), quandl.get("CHRIS/CBOE_VX2"), quandl.get("CHRIS/CBOE_VX3")]
    peRatios = []
    vixValues = [[], [], []]
    # divYields = []
    for date, row in closeData.iterrows():
        # PE Ratio
        peRatio = peRatioData["Value"][dateToYYYYMMDD(date, 1)]
        peRatios.append(peRatio)
        # DIV Yield
        # divYield = divYieldData["Value"][dateToYYYYMMDD(date, 1)]
        # divYields.append(divYield)
        for x in range(0, 3):
            if dateToYYYYMMDD(date) in vixData[x]["Open"]:
                vixValue = vixData[x]["Settle"][dateToYYYYMMDD(date)]
            else:
                vixValue = vixValues[x][len(vixValues[x]) - 1]
            vixValues[x].append(vixValue)

    closeData["peRatio"] = peRatios
    closeData["vix1"] = vixValues[0]
    closeData["vix2"] = vixValues[1]
    closeData["vix3"] = vixValues[2]

    closeData["vix1-3"] = closeData["vix1"] - closeData["vix3"]
    closeData["vix2-3"] = closeData["vix2"] - closeData["vix3"]
    closeData["vix1-2"] = closeData["vix1"] - closeData["vix2"]
    # closeData["divYield"] = divYields
    fig, axs = plt.subplots(2)
    axs[0].plot(closeData.loc[:, ["Close", "vix1"]], 'tab:blue')
    axs[0].set_title('SP500')
    axs[1].plot(closeData.loc[:, "peRatio"], 'tab:orange')
    axs[1].set_title('PE Ratio')
    # axs[2].plot(closeData.loc[:, "vix"], 'tab:green')
    # axs[2].set_title('Vol Price')
    # axs[2].plot(closeData.loc[:, "divYield"], 'tab:green')
    # axs[2].set_title('Dividend Yield')
    fig.show()
    return closeData

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

from keras.optimizers import SGD
# split into input (X) and output (y) variables
# "vix2", "vix3", "vix1-3", "vix2-3", "vix1-2"
trainingData = getDataset("^GSPC", 5)
trainX = trainingData.loc[:, ["sma50", "sma100", "sma200", "50-100", "50-200", "100-200", "spotTo50", "spotTo100", "spotTo200", "vix1", "peRatio"]].values
#print(X)
trainY = trainingData.loc[:, ["outcome"]].values
# define the keras model
model = Sequential()
model.add(Dense(20, input_dim=11, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(trainX, trainY, epochs=5, batch_size=5)
# evaluate the keras model
_, accuracy = model.evaluate(trainX, trainY)
#print('Accuracy: %.2f' % (accuracy*100))

testingDuration = 13
testingData = getDataset("^GSPC", testingDuration)
testingX = testingData.loc[:, ["sma50", "sma100", "sma200", "50-100", "50-200", "100-200", "spotTo50", "spotTo100", "spotTo200", "vix1", "peRatio"]].values

predictions = model.predict_classes(testingX)

#print(predictions)
testingGraph = testingData.loc[:, "Close"].to_frame()
print(testingGraph)
print(len(predictions))
testingGraph["pred"] = predictions
testingGraph["pred"] = testingGraph["pred"] * 50

currentValue = 100
current2XValue = 100
lastPrice = 0
portfolioValues = []
portfolioValues2X = []
for date, row in testingGraph.iterrows():
    if lastPrice == 0:
        lastPrice = row["Close"]
        portfolioValues.append(currentValue)
        portfolioValues2X.append(current2XValue)
        continue
    if row["pred"] != 0:
        dailyReturn = row["Close"] / lastPrice
        currentValue = currentValue * dailyReturn
        current2XValue = current2XValue * (((dailyReturn - 1) * 2) + 1)
        #print(currentValue)
        portfolioValues.append(currentValue)
        portfolioValues2X.append(current2XValue)
        lastPrice = row["Close"]
    elif SHORT:
        lastPrice = row["Close"]
        dailyReturn = row["Close"] / lastPrice
        currentValue = currentValue * (1 / dailyReturn)
        current2XValue = current2XValue * (1 / (((dailyReturn - 1) * 2) + 1))
        # print(currentValue)
        portfolioValues.append(currentValue)
        portfolioValues2X.append(current2XValue)
        lastPrice = row["Close"]
    else:
        lastPrice = 0
        portfolioValues.append(currentValue)
        portfolioValues2X.append(current2XValue)

testingGraph["portfolio"] = portfolioValues
#testingGraph['2x leverage'] = portfolioValues2X
#print(finalGraph)

#absolute Return calculation
underlyingStartingPrice = testingData.loc[:, "Close"][0]
underlyingLastPrice = testingData.loc[:, "Close"][len(testingData.loc[:, "Close"]) - 1]
underlyingReturns = (pow(underlyingLastPrice/underlyingStartingPrice, 1/(testingDuration)) - 1)*100
#print(underlyingReturns)
print(f'{bcolors.OKBLUE}Annual returns for Underlying: %.2f' % underlyingReturns + "%")

portfolioStartingPrice = 100
portfolioLastPrice = portfolioValues[len(portfolioValues) - 1]
portfolioReturns = (pow(portfolioLastPrice/portfolioStartingPrice, 1/(testingDuration)) - 1)*100
#print(portfolioReturns)
print('Annual returns for Portfolio: %.2f' % portfolioReturns + "%")

portfolio2XLastPrice = portfolioValues2X[len(portfolioValues2X) - 1]
portfolio2XReturns = (pow(portfolio2XLastPrice/portfolioStartingPrice, 1/(testingDuration)) - 1)*100
#print(portfolioReturns)
print('Annual returns for 2 times leveraged Portfolio: %.2f' % portfolio2XReturns + f"%{bcolors.ENDC}")


testingFig, testingAxes = plt.subplots(2)

testingAxes[0].plot(testingGraph.loc[:, ["portfolio"]], 'tab:blue', linewidth=0.3)
testingAxes[0].set_title('Portfolio')
testingAxes[1].plot(testingGraph.loc[:, ["pred", "Close"]], linewidth=0.3)
testingAxes[1].set_title('Underlying and Decisions')
testingFig.show()
testingFig.savefig('final.png', format='png', dpi=1000)
plt.show()

#for index, prediction in predictions:
#	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
