import argparse
from scipy.spatial.distance import cosine 
from scipy.spatial.distance import cdist 
import numpy as np
import os
import time
import datetime

from data_manager import DataManager
from ClassificationAndRegression.classification_and_regression_evaluation import Evaluator as Classification_Regression_evaluator
from Clustering.clustering_evaluation import Evaluator as Clustering_evaluator
from DocumentSimilarity.document_similarity_evaluation import Evaluator as Doc_Similarity_evaluator
from EntityRelatedness.entity_relatedness_evaluation import Evaluator as Entity_Relatedness_evaluator
from SemanticAnalogies.semantic_analogies_evaluation import Evaluator as Semantic_Analogies_evaluator

vectors_filename = ''
vectors_size = 0
distance_metric = ''
result_directory = ''
top_k = 2

def default_analogy_function(vec1, vec2, vec3):
    return (np.array(vec2) - np.array(vec1) + np.array(vec3))

analogy_function = default_analogy_function

def manage_parameters():
    parser = argparse.ArgumentParser(description='Evaluation framework for RDF embedding methods')
    parser.add_argument('--vectors_file', type=str, required=True, help='Path of the file where your vectors are stored. File format: one line for each entity with entity and vector')
    parser.add_argument('--vectors_length', default=200, type=int, help='Length of each vector. Default : 200')
    parser.add_argument('--distance_metric', default='cosine', help='Metric to measure the distance between two vectors. Default : cosine')
    parser.add_argument('--analogy_function', default=default_analogy_function, type=callable, help='Used in SemanticAnalogies : Metric to measure the analogy among vectors')
    parser.add_argument('--top_k', default=2, type=int, help='Used in SemanticAnalogies : The predicted vector will be compared with the top k closest vectors to establish if the prediction is correct or not. Default : 2')
    args = parser.parse_args()

    global vectors_filename
    global vectors_size
    global distance_metric
    global analogy_function
    global top_k

    vectors_filename = args.vectors_file
    vectors_size = args.vectors_length
    distance_metric = args.distance_metric
    analogy_function = args.analogy_function
    top_k = args.top_k

def create_result_directory():
    global result_directory
    result_directory = "results/result"+datetime.datetime.fromtimestamp(time.time()).strftime('_%Y-%m-%d_%H-%M-%S')

    try:  
        os.mkdir(result_directory)
    except OSError:  
        print ("Creation of the directory %s failed" % result_directory)

def run_tests():
    Classification_Regression_evaluator.evaluate(vectors, result_directory)
    Clustering_evaluator.evaluate(vectors, distance_metric, result_directory)
    Doc_Similarity_evaluator.evaluate(vectors, distance_metric, result_directory)
    Entity_Relatedness_evaluator.evaluate(vectors, distance_metric, result_directory)
    Semantic_Analogies_evaluator.evaluate(vectors, vectors_size, analogy_function, top_k, result_directory)

if __name__ == "__main__":
    manage_parameters()
    
    global vectors
    vectors = DataManager.read_vector_file(vectors_filename, vectors_size)
    print('Vectors read')

    create_result_directory()

    run_tests()






   