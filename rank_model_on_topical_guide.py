import json
import re
import pandas as pd
from calculate_similarity import createCosineSimilarityFile
from create_tensor import create_tensor
import torch
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, util
import sys


def parseCosineSimilarityFileTopicalGuide(similarityFile, topicalGuide, allEmbeddingDF, outputFile):

    topics = list(topicalGuide.keys())

    allEmbeddingDF = pd.read_csv(allEmbeddingDF)
    
    with open(outputFile,'w') as outFile:
        with open(similarityFile,'r') as inFile:
            for line in inFile:
                line = line.strip()
                if line.startswith("VerseIndex"):
                    outFile.write(line + "\n")
                    continue
                l = line.split("\t")

                topicIndex = int(l[0])

                small_embedding_ref = topics[topicIndex]
                outFile.write(f"{small_embedding_ref}\t")

                for verse in l[1:]:
                    verse = verse.split(":")
                    similarVerseIndex = int(verse[0])
                    similarity = float(verse[1])
                    all_embedding_ref = allEmbeddingDF.iloc[similarVerseIndex]["Reference"]
                    
                    outFile.write(f"{all_embedding_ref}: {similarity}\t")

                outFile.write("\n")



def topicalGuides_topics_similarity(topicalGuidePath, model_type, topicsEmbedding, allEmbedding, similarityFile, standardWorksDF, parsedSimilarityFile):
    with open(topicalGuidePath, "r") as readFile:
        topicalGuide = json.load(readFile)

    topics = list(topicalGuide.keys())

    create_tensor(model_type, topicsEmbedding, verses=topics)
    # STANDARD_WORKS_LEN = 41995

    longest_topic = max([len(topicalGuide[topic]) for topic in topics]) 

    createCosineSimilarityFile(topicsEmbedding,allEmbedding,similarityFile, longest_topic) #Takes a long time

    parseCosineSimilarityFileTopicalGuide(similarityFile, topicalGuide, standardWorksDF, parsedSimilarityFile) #Takes some time

def analyze_success(topicalGuideNoIndexPath, parsedSimilarityFile, outputFile):
    with open(topicalGuideNoIndexPath, "r") as readFile:
        topicalGuide = json.load(readFile)

    topicRankings = {}
    with open(parsedSimilarityFile, "r") as simFile:
        for line in simFile:
            if "VerseIndex" in line:
                continue
            l = line.split("\t")

            topicVerses = topicalGuide[l[0]]
            verseRank = []
            for i, verse in enumerate(l[1:]):
                parsed_verse = verse.split(": ")[0]
                if parsed_verse in topicVerses:
                    verseRank.append(i+1)
                    
            topicRankings[l[0]]= verseRank

    with open(outputFile,'w') as out:
        for topic, verseRanks in topicRankings.items():
            out.write(f'{topic}:{len(topicalGuide[topic])}\t')
            for verseRank in verseRanks:
                out.write(f'{verseRank}\t')
            out.write('\n')

    # for topic in topicalGuide:
    #     i = 0
    #     for topicName, topicVerses in topic.items():
    #         for verse, index in topicVerses.items():
                

    #         i+=1

def measure_accuracy(ranked_topical_guide):

    all_ranks = []
    grace_json = {}
    with open(ranked_topical_guide, 'r') as f:
        for line in f:
            l = line.strip().split('\t')
            topic = l[0].split(":")[0]
            topicLen = int(l[0].split(":")[1])
            ranks = l[1:]

            numRanksUnderK = [int(rank) for rank in ranks if int(rank) <= topicLen]
            grace_json[topic] = len(numRanksUnderK)/topicLen
            if len(numRanksUnderK) == 0:
                all_ranks.append(0)
            else:
                all_ranks.append(len(numRanksUnderK)/topicLen)

    with open('test_grace.json', 'w') as grace:
        json.dump(grace_json, grace)

    return sum(all_ranks)/len(all_ranks)


def rank_model(model_type, allEmbedding):
    standardWorksDF = "datasets/allWorks.csv"
    
    topicalGuideNoIndexPath = "topicalGuide/training_topical_guide.json"
    limitedTensorEmbeddingPath = allEmbedding.split("/")[1].strip(".pt")

    topicsEmbedding = f"modelTensors/{limitedTensorEmbeddingPath}_topics_training.pt"
    outputUnParsedFile = f"cosineSimilarity/unmapped/{limitedTensorEmbeddingPath}_topics_training.tsv"
    outputParsedFile = f"cosineSimilarity/mapped/{limitedTensorEmbeddingPath}_topics_training.tsv"

    topicalGuides_topics_similarity(topicalGuideNoIndexPath, model_type, topicsEmbedding, allEmbedding, outputUnParsedFile, standardWorksDF, outputParsedFile)

    ranked_topical_guide = f"analysis/{limitedTensorEmbeddingPath}_training_topics.tsv"

    analyze_success(topicalGuideNoIndexPath, outputParsedFile, ranked_topical_guide)
    
    recallAtKScore = measure_accuracy(ranked_topical_guide)  
    outputFile = "analysis/modelRankings"
    with open(outputFile, "a") as output:
        output.write(f"{allEmbedding}\ttraining\t{recallAtKScore}\n")
        
if __name__ == "__main__":
    topicalGuideNoIndexPath = "topicalGuide/training_topical_guide.json"
    standardWorksDF = "datasets/allWorks.csv"
    topicsEmbedding ='modelTensors/MiniLM_topicsTG_training.pt'


    # For Harlan's model
    topicsEmbedding ='modelTensors/MiniLM_topicsTG_training.pt'
    allEmbedding = 'modelTensors/allWorks_embedding.pt'
    #allEmbedding = 'modelTensors/MiniLM_allWorks_embedding_test.pt'
    similarityFile = "cosineSimilarity/unmapped/training_topicsAgainstAll.tsv"
    parsedSimilarityFile = "cosineSimilarity/mapped/training_topicsAgainstAll.tsv"
    model_type = 'all-MiniLM-L6-v2'
    # Creates the cosine similarity file for the topical guide topic names against all verses
    topicalGuides_topics_similarity(topicalGuideNoIndexPath, model_type, topicsEmbedding, allEmbedding, similarityFile, standardWorksDF, parsedSimilarityFile)
    

    ranked_topical_guide = "analysis/training_topicalGuideRankings.tsv"
    analyze_success(topicalGuideNoIndexPath, parsedSimilarityFile, ranked_topical_guide)
    print(measure_accuracy(ranked_topical_guide))

    # For Harlan's model with context
    # topicalGuideNoIndexPath = "topicalGuide/topical_guide_clean.json"
    # topicsEmbedding ='modelTensors/MiniLM_topicsTG.pt'
    # allEmbedding = 'modelTensors/MiniLM_allWorks_embedding_(1).pt'
    # similarityFile = "cosineSimilarity/unmapped/topicsAgainstAll_(1).tsv"
    # standardWorksDF = "standardWorksDF/allWorks.csv"
    # parsedSimilarityFile = "cosineSimilarity/mapped/topicsAgainstAll_(1).tsv"
    # model_type = 'all-MiniLM-L6-v2'
    # # Creates the cosine similarity file for the topical guide topic names against all verses
    # topicalGuides_topics_similarity(topicalGuideNoIndexPath, model_type, topicsEmbedding, allEmbedding, similarityFile, standardWorksDF, parsedSimilarityFile)
    

    # ranked_topical_guide = "analysis/topicalGuideRankings_(1).tsv"
    # analyze_success(topicalGuideNoIndexPath, parsedSimilarityFile, ranked_topical_guide)
    # print(measure_accuracy(ranked_topical_guide))


    # #For Drake's model
    # #topicalGuideNoIndexPath = "topicalGuide/topical_guide_clean.json"
    # model_type = 'sentence-transformers/all-roberta-large-v1'
    # topicsEmbedding ='modelTensors/roberta_topicsTG.pt'
    # allEmbedding = '../test_drake_model/drake_tensor.pt'
    # # allEmbedding = 'modelTensors/roberta-large_allWorks_embedding.pt'
    # #similarityFile = "cosineSimilarity/unmapped/topicsAgainstAll.tsv"
    # standardWorksDF = "standardWorksDF/allWorks.csv"
    # #parsedSimilarityFile = "cosineSimilarity/mapped/topicsAgainstAll.tsv"
    # similarityFile = 'cosineSimilarity/unmapped/topicsAgainstAll_drake_model.tsv'
    # parsedSimilarityFile = 'cosineSimilarity/mapped/topicsAgainstAll_drake_model.tsv'
    # #Creates the cosine similarity file for the topical guide topic names against all verses
    # topicalGuides_topics_similarity(topicalGuideNoIndexPath, model_type, topicsEmbedding, allEmbedding, similarityFile, standardWorksDF, parsedSimilarityFile)


    # ranked_topical_guide = "analysis/topicalGuideRankings_drake_model.tsv"
    # analyze_success(topicalGuideNoIndexPath, parsedSimilarityFile, ranked_topical_guide)
    # print(measure_accuracy(ranked_topical_guide))