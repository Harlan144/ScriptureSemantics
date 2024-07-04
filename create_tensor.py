#Create allWorksTensor.pt from allWorks.csv.
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import torch
from cv2 import getGaussianKernel 
# from transformers import AutoTokenizer,AutoModel
import sys


# Define a function to concatenate verses based on the given parameter x
def concatenate_verses(df, x):
    concatenated_verses = []
    for i, row in df.iterrows():
        conctatenated_verse = ""
        current_chapter = row["Chapter"]
        current_book = row["Book"]
        current_work= row["Work"]
        for j in range(i-x, i+x+1):
            if j < 0 or j >= len(df):
                continue
            if df.iloc[j]["Chapter"] == current_chapter and df.iloc[j]["Book"] == current_book and df.iloc[j]["Work"] == current_work:
                conctatenated_verse+= df.iloc[j]["Verse"] +" "

        concatenated_verses.append(conctatenated_verse)       

    return concatenated_verses

# Define a function to calculate the weighted average of the embeddings
def weighted_average(embeddings, mask_weights, weights):
    total_weight = sum([mask*weight[0] for mask, weight in zip(mask_weights, weights)])
    normalizer = 1/total_weight

    updated_weights= np.array([weight[0]*normalizer*mask for  mask, weight in zip(mask_weights, weights)])
    embeddings_array = np.stack(embeddings)
    return torch.from_numpy(np.average(embeddings_array, weights=updated_weights, axis=0))

# Define a function to add context to the embeddings using a gaussian kernel
def add_context_gaussian(embeddings, df, context_window):
    weights = getGaussianKernel(context_window*2+1, 1)
    embedding_dim = len(embeddings[0])

    len_df = len(df)
    context_embeddings = []

    for i, row in df.iterrows():
        conctatenated_embedding = []
        current_chapter = row["Chapter"]
        current_book = row["Book"]
        current_work= row["Work"]
        mask_weights = []
        for j in range(i-context_window, i+context_window+1):
            if j < 0 or j >= len_df:
                mask_weights.append(0)
                conctatenated_embedding.append(torch.zeros(embedding_dim)) #check this
                continue
            if df.iloc[j]["Chapter"] == current_chapter and df.iloc[j]["Book"] == current_book and df.iloc[j]["Work"] == current_work:
                conctatenated_embedding.append(embeddings[j])
                mask_weights.append(1)
            else:
                mask_weights.append(0)
                conctatenated_embedding.append(torch.zeros(embedding_dim))

        averaged_embedding = weighted_average(conctatenated_embedding, mask_weights, weights)
        context_embeddings.append(averaged_embedding)       

    return torch.stack(context_embeddings).float()

# Define a function to embed the verses using the given model and save the embeddings to a tensor
def create_tensor(model_type, tensor_path, df_path='datasets/allWorks.csv', verses='', context_window=0):
    model = SentenceTransformer(model_type)
    tokenizer = model.tokenizer
    max_seq_length = model.max_seq_length
    if not verses:
        df = pd.read_csv(df_path)
        verses = list(df["Verse"])

    if context_window!=0:
        if model_type == 'sentence-transformers/all-roberta-large-v1':
            embeddings = torch.load(f'modelTensors/all-roberta-large-v1_with_punc.pt')
        else:
            embeddings= torch.load(f'modelTensors/{model_type}_with_punc.pt')
        contextual_embeddings = add_context_gaussian(embeddings,df, context_window)
        torch.save(contextual_embeddings, tensor_path)

    else: 
        print('found verses')

        encoded_verses = [tokenizer.encode(verse) for verse in verses]
        verses_over_max = [(index, verse) for index, verse in enumerate(encoded_verses) if len(verse) > max_seq_length]
        for index, verse in verses_over_max:
            print(f"Verse {index} is too long: {len(verse)}")


        allEmbeddings = model.encode(verses, convert_to_tensor=True, show_progress_bar=True)
        torch.save(allEmbeddings, tensor_path)


if __name__ == '__main__':
    model_type = sys.argv[1]
    context_window = sys.argv[2]
    depth = sys.argv[3]
    include_punc = True
    df_path='datasets/allWorks.csv'

    try:
        int(context_window)
    except:
        print("Error! Context window is not an integer")
    try:
        int(depth)
    except:
        print("Error! Depth is not an integer")
    

    #Determine output_tensor_path
    if "/" in model_type:
        model_name = model_type.split("/")[1]
        output_tensor_path = f'modelTensors/{model_name}'
    else:
        output_tensor_path = f'modelTensors/{model_type}'

    if context_window!="0":
        output_tensor_path+=f"_({context_window})"
    if include_punc:
        output_tensor_path+="_with_punc"
    output_tensor_path+=".pt"


    create_tensor(model_type, output_tensor_path, context_window=int(context_window), df_path=df_path)
    from calculate_similarity import create_similarity_files
    from rank_model_on_topical_guide import rank_model
    
    create_similarity_files(output_tensor_path, depth)
    rank_model(model_type, output_tensor_path)
    # #create_tensor('all-MiniLM-L6-v2', 'modelTensors/MiniLM_allWorks_embedding_with_punc.pt',context_window=0, df_path='standardWorksDF/allWorks.csv')
    # #create_tensor('all-MiniLM-L6-v2', 'modelTensors/MiniLM_allWorks_embedding_(2).pt', context_window =2, df_path='standardWorksDF/allWorks.csv')
    # #create_tensor('sentence-transformers/all-roberta-large-v1', 'modelTensors/roberta-large_allWorks_embedding_with_punc.pt', context_window = 0, df_path='standardWorksDF/allWorks.csv')
    # create_tensor('sentence-transformers/all-roberta-large-v1', 'modelTensors/roberta-large_allWorks_embedding_(1).pt', context_window = 1, df_path='standardWorksDF/allWorks.csv')
    # create_tensor('sentence-transformers/all-roberta-large-v1', 'modelTensors/roberta-large_allWorks_embedding_(2).pt', context_window = 2, df_path='standardWorksDF/allWorks.csv')
