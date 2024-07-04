# This file is used to calculate the cosine similarity between the verses of different works
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import torch
import heapq
import sys

#Use pytorch to open the tensor files (NOT ON GITHUB- too large, please ask for them)
#and use those to find the cosine similarity for each verse of the BOM to all other standard works.
#In a seperate file (similarityAnalysisToBible/parseBOMSimilarityVerses.py),
# we will limit the similar verses.


def createCosineSimilarityFile(smallEmbeddingFile,allEmbeddingFile,outputFile, similarVerseCount=15):

    allEmbedding = torch.load(allEmbeddingFile)
    smallEmbedding = torch.load(smallEmbeddingFile)
    # print(smallEmbedding.shape)
    # print(allEmbedding.shape)
    cosine_scores = util.cos_sim(smallEmbedding, allEmbedding)

    # print(cosine_scores.shape) #These should match
    # print(len(smallEmbedding), len(allEmbedding))
    
    #Find the top similarVerseCount verses for each verse.
    top_results = torch.topk(cosine_scores, k=similarVerseCount)

    #Output the results to a file
    with open(outputFile,'w') as outFile:
        i  = 0    
        outFile.write("VerseIndex\tSimilarVerseIndex: CosineSimilarity\n")
        for score, idx in zip(top_results[0], top_results[1]):
            outFile.write(f"{i}\t")
            for j in range(len(idx)):
                if idx[j] != i: #Don't include the verse itself
                    outFile.write(f"{idx[j]}: {score[j]:.2f}\t")
            outFile.write("\n")      
            i+=1

#Parse the cosine similarity file to include the references of the verses
def parseCosineSimilarityFile(similarityFile, smallEmbeddingDF, allEmbeddingDF, outputFile):

    smallEmbeddingDF = pd.read_csv(smallEmbeddingDF)
    allEmbeddingDF = pd.read_csv(allEmbeddingDF)
    
    with open(outputFile,'w') as outFile:
        with open(similarityFile,'r') as inFile:
            for line in inFile:
                line = line.strip()
                if line.startswith("VerseIndex"):
                    outFile.write(line + "\n")
                    continue
                l = line.split("\t")

                verseIndex = int(l[0])
                small_embedding_ref = smallEmbeddingDF.iloc[verseIndex]["Reference"]
                outFile.write(f"{small_embedding_ref}\t")

                for verse in l[1:]:
                    verse = verse.split(":")
                    similarVerseIndex = int(verse[0])
                    similarity = float(verse[1])
                    all_embedding_ref = allEmbeddingDF.iloc[similarVerseIndex]["Reference"]
                    
                    if similarVerseIndex!= verseIndex: #Don't include the verse itself
                        outFile.write(f"{all_embedding_ref}: {similarity}\t")

                outFile.write("\n")

#Create the similarity files for the different embeddings
def create_similarity_files(tensorEmbeddingPath, depth=25):
    standardWorksDF = "datasets/allWorks.csv"

    limitedTensorEmbeddingPath = tensorEmbeddingPath.split("/")[1].strip(".pt")

    outputUnParsedFile = f"cosineSimilarity/unmapped/{limitedTensorEmbeddingPath}.tsv"
    outputParsedFile = f"cosineSimilarity/mapped/{limitedTensorEmbeddingPath}.tsv"

    createCosineSimilarityFile(tensorEmbeddingPath,tensorEmbeddingPath,outputUnParsedFile, int(depth))
    parseCosineSimilarityFile(outputUnParsedFile,standardWorksDF,standardWorksDF,outputParsedFile)

if __name__ == '__main__':
    
    outputFile = "cosineSimilarity/unmapped/allSimilarVersesDepth25_with_punc"
    allEmbedding = "modelTensors/MiniLM_allWorks_embedding_with_punc.pt"
    createCosineSimilarityFile(allEmbedding,allEmbedding,outputFile, 25)

    standardWorksDF = "datasets/allWorks.csv"
    parseCosineSimilarityFile(outputFile,standardWorksDF,standardWorksDF,"cosineSimilarity/mapped/allSimilarVersesDepth25_with_punc")