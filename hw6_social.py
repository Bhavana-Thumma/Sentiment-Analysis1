"""
15-110 Hw6 - Social Media Analytics Project
Name:
AndrewID:
"""

import hw6_social_tests as test

project = "Social" # don't edit this

### WEEK 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    df = pd.read_csv (filename)
    return df


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    l = fromString.split()
    n = ""
    for i in l:
        if(i=='From:'):
            continue
        elif(i.startswith("(")):
            break
        else:
            n+=i+" "
    # print(n[:-1])
    return n[:-1]


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    l = fromString.split("(")
    n = []
    s = l[1].split()
    for i in s:
        if(i=="from"):
            break
        else:
            n.append(i)
    return " ".join(n)


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    l = fromString.split("from ")
    # print(l[1][:-1])
    return "".join(l[1][:-1])


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    fl = [ ]
    if("#" not in message):
        return fl
    l = message.split("#")
    for i in l[1:]:
        w = ''
        for j in i:
            # "[@_!$%^&*()<>?/\|}{~:].;,' "
            if(j in endChars):
                break
            else:
                w+=j
        w="#"+w
        fl.append(w)
    # print(fl)
    return fl        


'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    r = ""
    for i in range(len(stateDf['state'])):
        if(stateDf['state'][i] == state):
            r = stateDf['region'][i]
    # print(r)
    return r


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    state = []
    name = []
    position= []
    region = []
    hashtags= []
    for i in range(len(data)):
        state.append(parseState(data['label'][i]))
        name.append(parseName(data['label'][i]))
        position.append(parsePosition(data['label'][i]))
        region.append(getRegionFromState(stateDf, parseState(data['label'][i])))
        hashtags.append(findHashtags(data['text'][i]))
    data['name'] = name
    data['state'] = state
    data['region'] = region
    data['position'] = position
    data['hashtags'] = hashtags

    return None


### WEEK 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if(score < -0.1):
        return "negative"
    elif(score > 0.1):
        return "positive"
    return "neutral"


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    l = []
    for i in range(len(data)):
        l.append(findSentiment(classifier, data['text'][i]))
    data['sentiment'] = l
    return None


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    d = {}
    for i in range(len(data)):
        if(colName!="" and dataToCount!=""):
            if(data[colName][i]== dataToCount and data['state'][i] in d):
                d[data['state'][i]]+=1
            elif(data[colName][i]== dataToCount and data['state'][i] not in d):
                d[data['state'][i]]=1
        else:
            if(data['state'][i] in d):
                d[data['state'][i]]+=1
            elif(data['state'][i] not in d):
                d[data['state'][i]]=1
    # print(len(d))
    return d


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    d = {}
    for rgn in set(data['region']):
        d[rgn] = {}
        for j in range(len(data)):
            c = data[colName][j]
            r = data['region'][j]
            if(r == rgn and c not in d[rgn]):
                d[rgn][c] = 1
            elif(r == rgn and c in d[rgn]):
                d[rgn][c] += 1
    # print(d)
    return d


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    d = {}
    for i in data['hashtags']:
        for j in i:
            if(j not in d):
                d[j] = 1
            else:
                d[j]+=1
    return d


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    d = hashtags
    d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
    c = 0
    d1 = {}
    for i in d:
        d1[i] = d[i]
        c+=1
        if(c == count):
            break
    # print(d1)
    return d1

    # vl = list(d.values())
    # d1 = {}
    # i = 0
    # while(True):
    #     for k, v in d.items():
    #         if(vl[i] == v):
    #             d1[k] = v
    #         if(len(d1)==count):
    #             return d1       
    #     i+=1


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    c = 0
    sc = 0
    for i in range(len(data)):
        if(hashtag in data['hashtags'][i]):
            c+=1
            if(data['sentiment'][i]=='positive'):
                sc+=1
            elif(data['sentiment'][i]=='negative'):
                sc+=-1
            else:
                continue    
    # print(sc/c)    
    return sc/c


### WEEK 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    states = list(stateCounts.keys())
    counts = list(stateCounts.values())
    # width = 0.8
    # fig, ax = plt.subplots()
    # rects1 = ax.bar(states, counts, width, color='r')

    # ax.set_ylim(0,600)
    # ax.set_ylabel('Frequency')
    # ax.set_title(title)
    # ax.set_xticklabels(states)

    # def autolabel(rects):
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
    #                 '%d' % int(height),
    #                 ha='center', va='bottom')
    # plt.xticks(rotation=90)
    # autolabel(rects1)

    # plt.show()
    plt.figure(figsize=(10,5))
    plt.bar(states, counts)
    # Rotate the name of sports by 90 degree in x-axis
    plt.xticks(rotation=90)
    # show the graph 
    plt.title(title)
    plt.show()
    # return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    import matplotlib.pyplot as plt

    for k, v in stateFeatureCounts.items():
        stateFeatureCounts[k] /= stateCounts[k]
    # print(stateFeatureCounts)
    d = dict(sorted(stateFeatureCounts.items(), key=lambda item: item[1], reverse=True))
    # print(stateFeatureCounts)
    c = 0
    d1 = {}
    for i in d:
        d1[i] = d[i]
        c+=1
        if(c == n):
            break
    
    states = list(d1.keys())
    counts = list(d1.values())

    plt.figure(figsize=(10,5))
    plt.bar(states, counts)
    # Rotate the name of sports by 90 degree in x-axis
    plt.xticks(rotation=90)
    # show the graph 
    plt.title(title)
    plt.show()
    # return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    fl = []
    region = []
    frl = []
    for i in regionDicts:
        l = []
        dic = regionDicts[i]
        for j in dic:
            if(j not in fl):
                fl.append(j)
            l.append(dic[j])          
        frl.append(l)
        region.append(i)
    # print(fl)
    # print(region)
    # print(frl)
    sideBySideBarPlots(fl, region, frl, title)


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    d = mostCommonHashtags(getHashtagRates(data), 50)
    h = []
    hv = []
    s = []
    for k in d:
        if(k not in h):
            h.append(k)
            hv.append(d[k])
            s.append(getHashtagSentiment(data, k))
    # print(h)
    # print(hv)
    # print(s)
    scatterPlot(hv, s, h, 'title')



#### WEEK 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)
    
    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    # Uncomment these for Week 2 ##
    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()