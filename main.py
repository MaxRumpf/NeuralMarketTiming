
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

msft = yf.Ticker("^GSPC")
numYears = 10
dataset = msft.history(period=str(numYears)+"y")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

#closeData.plot(linewidth=0.3)

#for x in range(2000, 3500, 250):
 #   plt.axhline(y=x, linewidth=0.2, linestyle='--')




from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

from keras.optimizers import SGD
# split into input (X) and output (y) variables
X = closeData.loc[:, ["Close", "sma50", "sma100", "sma200", "50-100", "50-200", "100-200", "spotTo50", "spotTo100", "spotTo200"]].values
#print(X)
y = closeData.loc[:, ["outcome"]].values
# define the keras model
model = Sequential()
model.add(Dense(20, input_dim=10, activation='relu'))
#model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=5, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
#print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict_classes(X)

#print(predictions)
finalGraph = closeData.loc[:, "Close"].to_frame()
finalGraph["pred"] = predictions
finalGraph["pred"] = finalGraph["pred"] * 50

currentValue = 100
lastPrice = 0
portfolioValues = []
for date, row in finalGraph.iterrows():
    if lastPrice == 0:
        lastPrice = row["Close"]
        portfolioValues.append(currentValue)
        continue
    if row["pred"] != 0:
        dailyReturn = row["Close"] / lastPrice
        currentValue = currentValue * dailyReturn
        #print(currentValue)
        portfolioValues.append(currentValue)
        lastPrice = row["Close"]
    else:
        lastPrice = 0
        portfolioValues.append(currentValue)

finalGraph["portfolio"] = portfolioValues
#print(finalGraph)

#absolute Return calculation
underlyingStartingPrice = closeData.loc[:, "Close"][0]
underlyingLastPrice = closeData.loc[:, "Close"][len(closeData.loc[:, "Close"]) - 1]
underlyingReturns = (pow(underlyingLastPrice/underlyingStartingPrice, 1/(numYears)) - 1)*100
#print(underlyingReturns)
print(f'{bcolors.OKBLUE}Annual returns for Underlying: %.2f' % underlyingReturns + "%")

portfolioStartingPrice = 100
portfolioLastPrice = portfolioValues[len(portfolioValues) - 1]
portfolioReturns = (pow(portfolioLastPrice/portfolioStartingPrice, 1/(numYears)) - 1)*100
#print(portfolioReturns)
print('Annual returns for Portfolio: %.2f' % portfolioReturns + f"%{bcolors.ENDC}")

finalGraph.plot(linewidth=0.3)
plt.savefig('final.png', format='png', dpi=1000)
plt.show()

#for index, prediction in predictions:
#	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
