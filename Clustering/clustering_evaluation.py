from sklearn import metrics
from sklearn import datasets, linear_model
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
from collections import defaultdict
import clustering_data_manager as data_manager
import clustering_model as model
import csv

class Evaluator:
    def __init__(self):
        print("Clustering evaluator init")

    @staticmethod
    def evaluate(vectors, distance_metric, results_folder):
        gold_standard_filenames = ['citiesAndCountries_cluster', 'cities2000AndCountries_cluster', 'citiesMoviesAlbumsCompaniesUni_cluster', 'teams_cluster']
        n_clusters_list = [2, 2, 5, 2]

        for i in range(len(gold_standard_filenames)):
            gold_standard_filename = gold_standard_filenames[i]
            gold_standard_file = 'Clustering/data/'+gold_standard_filename+'.txt'
            n_clusters = n_clusters_list[i]

            #clustering_models = ["DB", "KMeans", "AC", "WHC", "SC"]
            clustering_models = ["DB", "KMeans", "AC", "WHC"]

            data, ignored = data_manager.DataManager.read_data(vectors, gold_standard_file)
            
            if vectors.size == 0:
                print('Empty vector list')
            elif data.size == 0:
                print('Problems in merging vectors with gold standards ' + gold_standard_filename)
            else:
                print('Ignored data : ' + str(len(ignored)))

                scores = defaultdict(list)
                
                for model_name in clustering_models:
                    clustering_model = model.Model(model_name, metric = distance_metric, n_clusters=n_clusters)
                    scores[model_name].append(clustering_model.train(data, ignored))

            with open(results_folder+'/clustering_'+gold_standard_filename+'.csv', "wb") as csv_file:
                fieldnames = ['task_name', 'model_name', 'model_configuration', 'num_clusters', 'adjusted_rand_index', 'adjusted_mutual_info_score', 
            'fowlkes_mallows_score', 'homogeneity_score', 
            'completeness_score', 'v_measure_score']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for (method, scoresForMethod) in scores.items():
                    for score in scoresForMethod:
                        writer.writerow(score)