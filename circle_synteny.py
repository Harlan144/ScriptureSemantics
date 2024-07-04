# Description: This script generates a synteny plot of the scriptural books and chapters based on the cosine similarity between them.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.path as mpath
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
import math

###EXAMPLE INPUT###
#Use a reference file that is a tsv of each chapter/book and what chapter/book it links to.
# allWorks = "standardWorksDF/allWorks.csv"
# parsedReferenceFile="chapterTokenTensor/allBookSimilarity0.65"
# outputSyntenyPlot= "chapterTokenTensor/allBookSimilarity0dot65"

# Defines the white space between different works in the circle
DEGREE_BETWEEN_WORKS = 2

# Names of the works in our tsv file
WORKNAMES = {"pearl-of-great-price":"Pearl of Great Price",
                "old-testament": "Old Testament","book-of-mormon":"Book of Mormon",
                "d&c":"D&C", "new-testament":"New Testament"
}

# Function to find the x and y coordinates of a point on a circle given an angle and radius. Works counter-clockwise
def findPosBasedOnAngle(angle, radius): 
    radianAngle= math.radians(angle)
    x=math.cos(radianAngle)*radius
    y= math.sin(radianAngle)*radius
    return x,y

# Function to calculate the indexes of the works in the dataframe
def calculate_work_indexes(allWorksDF, worksAndLens):
    workIndexes ={}
    for work,length in worksAndLens.items():
        #only get the first?
        firstIndex =allWorksDF.loc[allWorksDF["Work"]==work].index[0]
        lastIndex = firstIndex+length #NOT INCLUSIVE
        workIndexes[work]= (firstIndex,lastIndex)
    return workIndexes

# Function to calculate the angles of the works in the circle based on the index
def calculate_work_angles(workIndexes, anglePerVerse):
    currentAngle =0
    workAngles= {}
    for work, indexRange in workIndexes.items():
        endAngle = currentAngle+(indexRange[1]-indexRange[0])*anglePerVerse
        workAngles[work]= (currentAngle,endAngle)
        currentAngle =endAngle+DEGREE_BETWEEN_WORKS

    return workAngles

# Function to find the books in the works that have more than minBookLen verses.
# This are colored gray and the names are omitted for clarity.
def find_books_in_works(allWorksDF, workIndexes, minBookLen):
    totalBooks = 0
    booksInWorks = {}

    for work in workIndexes:
        books= allWorksDF[allWorksDF["Work"]==work]["Book"].value_counts(sort=False).loc[lambda x:x>minBookLen]
        limitedBooks = list(books.index)
        totalBooks+=len(limitedBooks)
        booksInWorks[work]=limitedBooks

    return booksInWorks, totalBooks

# Function to calculate the angles of the books in the circle based on the index
def calculate_book_angles(booksInWorks, workAngles, workIndexes, allWorksDF):
    bookAngles= {}
    for work,bookList in booksInWorks.items():
        angleOfWork = workAngles[work]
        totalAngle = angleOfWork[1]-angleOfWork[0]
        indexOfWork = workIndexes[work]
        workLen = indexOfWork[1]-indexOfWork[0]
        for book in bookList:
            bookDF =allWorksDF.loc[allWorksDF["Book"]==book].index
            firstIndex = bookDF[0]
            lastIndex = bookDF[-1]
            startingAngle = (firstIndex-indexOfWork[0])/workLen*totalAngle+angleOfWork[0]
            endingAngle=(lastIndex-indexOfWork[0])/workLen*totalAngle+angleOfWork[0]
            if work in bookAngles:
                bookAngles[work][book] = (startingAngle,endingAngle)
            else:
                bookAngles[work]= {book:(startingAngle,endingAngle)}

    return bookAngles

# Function to find the x and y coordinates of a point on a circle given a verse number and radius.
def findPosFromVerseNum(verseNum, workIndexes, workAngles, circleRadius):
    for work,workRange in workIndexes.items():
        if verseNum>=workRange[0] and verseNum<workRange[1]:
            #it is in that book.
            workLen =workRange[1]-workRange[0]
            workAngleRange = workAngles[work]
            angleTotal = workAngleRange[1]-workAngleRange[0]
            fractionThroughWork = (verseNum-workRange[0])/workLen

            verseAngle= workAngleRange[0]+fractionThroughWork*angleTotal
            versePoint= findPosBasedOnAngle(verseAngle,circleRadius*0.975)
            return versePoint

# Function to graph the circle with the works and books
def graph_circle(totalVerseCount, cosineThreshold, totalBooks, workAngles, bookAngles):
    centerOfGraph = (0,0)
    circleDiameter= totalVerseCount/math.pi #not exact.
    circleRadius =circleDiameter/2

    #Add a white rectangle as a backround
    centerX= 0
    rectOnTop = plt.Rectangle((0,circleRadius*1.8), 10, 10, fc='white',ec="white")
    plt.gca().add_patch(rectOnTop)

    #Add the title
    titleText= "Synteny Analysis of Similarity between Different Scriptural Books"
    plt.text(centerX,1.7*(circleRadius), titleText, ha='center', rotation=0, wrap=True, fontsize=15)

    #Add the subtitle
    if cosineThreshold!=0:
        subTitleText = f"Calculated At {cosineThreshold:.2f} cosine similarity threshold"
        plt.text(centerX,1.5*(circleRadius), subTitleText, ha='center', rotation=0, wrap=True, fontsize=8)

    #Add the works
    color = iter(cm.rainbow(np.linspace(0, 1, totalBooks)))
    for work, angles in workAngles.items():
        arc= mpatch.Arc(centerOfGraph,
                width = circleDiameter,
                height= circleDiameter,
                angle= 0,
                theta1= angles[0],
                theta2= angles[1],
                linewidth=5,
                ec="gray"
        )
        plt.gca().add_patch(arc)
        textAngle = (angles[1]+angles[0])/2
        textX, textY= findPosBasedOnAngle(textAngle,circleRadius*1.32)

        #Fix the text rotation so it doesn't overlap
        if textAngle<180:#OLD TEST
            plt.text(textX+500, textY, WORKNAMES[work], ha='center', rotation=0, wrap=True, fontsize=7)
        elif textAngle>350: #POGP
            plt.text(textX+1500, textY, WORKNAMES[work], ha='center', rotation=0, wrap=True, fontsize=7)
        elif textAngle>180 and textAngle<250: #NT
            plt.text(textX-600, textY, WORKNAMES[work], ha='center', rotation=0, wrap=True,fontsize=7)
        else:
            plt.text(textX, textY, WORKNAMES[work], ha='center', rotation=0, wrap=True,fontsize=7)

    #Add the books
    for work, bookDic in bookAngles.items():
            for book, bookAngles in bookDic.items():
                c = next(color)

                arc= mpatch.Arc(centerOfGraph,
                        width = circleDiameter,
                        height= circleDiameter,
                        angle= 0,
                        theta1= bookAngles[0],
                        theta2= bookAngles[1],
                        linewidth=5,
                        ec=c
                )
                plt.gca().add_patch(arc)
                textAngle = (bookAngles[1]+bookAngles[0])/2
                textX, textY= findPosBasedOnAngle(textAngle,circleRadius*1.03)
                plt.text(textX, textY, book, ha='left', rotation=textAngle, wrap=True, fontsize=3)

# Function to check if two verses are in the same work
def in_same_work(verse1, verse2, workIndexes):
    for __, workRange in workIndexes.items():
        if verse1>=workRange[0] and verse1<workRange[1]:
            if verse2>=workRange[0] and verse2<workRange[1]:
                return True
    return False

# Function to parse the reference file to include the references of the verses
def parse_reference_file(cosineSimilarityFile, parsedReferenceFile, workIndexes, cosineThreshold, includeOnlyFirst):
    outputLines = []
    with open(cosineSimilarityFile) as similarityFile:
        index = 0

        for line in similarityFile:
            outputLine= ""
            if "VerseIndex" in line:
                continue
            l = line.strip().split("\t")

            x = len(l)
            if includeOnlyFirst:
                x=2
            for versePair in l[1:x]:
                secondVerse= versePair.split(": ")
                secondVerseNum = int(secondVerse[0])
                secondVerseCosine = float(secondVerse[1])
                if secondVerseCosine>cosineThreshold and not in_same_work(secondVerseNum, index, workIndexes):
                    outputLine+= f"\t{secondVerseNum}"

            if outputLine:
                outputLine = f"{index}{outputLine}\n"
                outputLines.append(outputLine)

            index+=1

    with open(parsedReferenceFile, "w") as file:
        for line in outputLines:
            file.write(line)
    return 

# Function to graph the lines between the verses. Uses bezier curves.
def graph_lines(parsedReferenceFile, totalVerseCount, workIndexes, workAngles):
    Path= mpath.Path

    circleRadius = totalVerseCount/math.pi/2 #not exact.
    with open(parsedReferenceFile) as file:
        for line in file:
            l = line.strip().split("\t")
            verseNum = int(l[0])
            versePoint = findPosFromVerseNum(verseNum, workIndexes, workAngles, circleRadius)

            for secondVerse in l[1:]:
                secondVerseNum = int(secondVerse)

                verse2Point = findPosFromVerseNum(int(secondVerseNum), workIndexes, workAngles, circleRadius) 
                bezierCurve = mpatch.PathPatch(Path([versePoint,(0,0),verse2Point],
                                                    [Path.MOVETO,Path.CURVE3,Path.CURVE3]),
                                                    fc="none", linewidth=0.1)
                                                
                plt.gca().add_patch(bezierCurve)

#Main function that graphs the circle synteny plot.  
def circleSyntenyGraph(allWorks,cosineSimilarityFile, outputSyntenyPlot, parsedReferenceFile, includeOnlyFirst =False,minBookLen=200,cosineThreshold=0):

    allWorksDF= pd.read_csv(allWorks)

    worksAndLens = allWorksDF["Work"].value_counts(sort=False)

    totalWorkCount = len(worksAndLens)
    totalVerseCount = len(allWorksDF)
    degreeBetweenWorks = DEGREE_BETWEEN_WORKS


    anglePerVerse= (360-degreeBetweenWorks*totalWorkCount)/(totalVerseCount) 

    workIndexes = calculate_work_indexes(allWorksDF, worksAndLens)

    workAngles = calculate_work_angles(workIndexes, anglePerVerse)

    booksInWorks, totalBooks = find_books_in_works(allWorksDF, workIndexes, minBookLen)

    bookAngles = calculate_book_angles(booksInWorks, workAngles, workIndexes, allWorksDF)


    plt.axes()
    graph_circle(totalVerseCount, cosineThreshold, totalBooks, workAngles, bookAngles)
    print('graphed circle')


    parse_reference_file(cosineSimilarityFile, parsedReferenceFile, workIndexes, cosineThreshold, includeOnlyFirst)

    graph_lines(parsedReferenceFile, totalVerseCount, workIndexes, workAngles)
    print('graphed lines')

    plt.axis('scaled')
    plt.axis('off')
    plt.savefig(outputSyntenyPlot,dpi=1200)

if __name__ == "__main__":

    allWorksPath = "datasets/allWorks.csv"
    cosineSimilarityFile="cosineSimilarity/unmapped/all-MiniLM-L6-v2_with_punc.tsv"
    parsedReferenceFile = 'cosineSimilarity/mapped/all-MiniLM-L6-v2_with_punc.tsv'

    outputSyntenyPlot= "syntenyPlots/allBook0dot85_with_punc.png"
    circleSyntenyGraph(allWorksPath,cosineSimilarityFile,outputSyntenyPlot, parsedReferenceFile, includeOnlyFirst=False,minBookLen=200,cosineThreshold=0.85)


