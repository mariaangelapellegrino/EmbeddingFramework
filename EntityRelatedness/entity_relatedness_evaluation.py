import pandas as pd
import numpy as np
import entity_relatedness_data_manager as data_manager
import entity_relatedness_model as model
import csv

class Evaluator:
    def __init__(self):
        print("Entity relatedness evaluator init")

    @staticmethod
    def evaluate(vectors, distance_metric, results_folder):
        gold_standard_filename = "EntityRelatedness/data/KORE.txt"
        entities, groups = data_manager.DataManager.read_gold_standard_file(gold_standard_filename)

        entities_df = pd.DataFrame(list(entities), columns = ['name'])
        #data, ignored = data_manager.DataManager.merge_data(vectors, entities_df)

        scores = list()

        left_entities_df = pd.DataFrame({'name':groups.keys()})
        left_merged, left_ignored = data_manager.DataManager.merge_data(vectors, left_entities_df)

        #print(left_merged)

        if left_merged.size == 0:
            print('EntityRelatedeness : no left entities of KORE in vectors')
        else:
            right_merged_list = list()
            right_ignored_list = list()
        
            for key in groups.keys():
                right_entities_df = pd.DataFrame({'name': groups[key]})
                right_merged, right_ignored = data_manager.DataManager.merge_data(vectors, right_entities_df)
                right_merged_list.append(right_merged)
                right_ignored_list.append(right_ignored)
                #print(right_merged)

            predicted_rank_list = model.Model.compute_relatedness(left_merged, left_ignored, right_merged_list, right_ignored_list, distance_metric)
            gold_rank_list = np.tile(np.arange(1, 21), (21, 1))
            scores = model.Model.evaluate_ranking(groups.keys(), gold_rank_list, predicted_rank_list)

        with open(results_folder+'/entityRelatedness_KORE.csv', "wb") as csv_file:
            fieldnames = ['task_name', 'gold_standard_file', 'entity_name', 'kendalltau_correlation', 'kendalltau_pvalue']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for score in scores:
                writer.writerow(score)