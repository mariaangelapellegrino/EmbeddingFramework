import numpy as np
import csv
import semantic_analogies_data_manager as data_manager
import semantic_analogies_model as model

class Evaluator:
    def __init__(self):
        print('SemanticAnalogies evaluator initialized')

    @staticmethod
    def evaluate(vectors, vec_size, analogy_function, top_k, results_folder):
        vocab = data_manager.DataManager.create_vocab(vectors)
        W_norm = model.Model.normalize_vectors(vectors, vec_size, vocab)

        gold_standard_filenames = ['capital_country_entities', 'all_capital_country_entities',
            'currency_entities', 'city_state_entities']

        scores = list()

        for gold_standard_filename in gold_standard_filenames:
            gold_standard_file = 'SemanticAnalogies/data/'+gold_standard_filename+'.txt'

            data = data_manager.DataManager.read_data(vocab, gold_standard_file)
            
            if len(data) == 0:
                print('SemanticAnalogies : Problems in merging vector with gold standard ' + gold_standard_file)
            else:
                right_answers, tot_answers, accuracy = model.Model.compute_semantic_analogies(vocab, data, W_norm, analogy_function, top_k)
                scores.append({'task_name':'Semantic Analogies', 'top_k_value':top_k, 'right_answers':right_answers, 'tot_answers':tot_answers, 'accuracy':accuracy})

            with open(results_folder+'/semanticAnalogies_'+gold_standard_filename+'.csv', 'wb') as file_result:
                fieldnames = ['task_name', 'top_k_value', 'right_answers', 'tot_answers', 'accuracy']
                writer = csv.DictWriter(file_result, fieldnames=fieldnames)
                writer.writeheader()
                for score in scores:
                    writer.writerow(score)

