class DataManager:
    def __init__(self):
        print('SemanticAnalogies data manager initialized')

    @staticmethod
    def create_vocab(vectors):
        words = vectors['name']
        vocab_size = len(words)
        vocab = {w: idx for idx, w in enumerate(words)}

        return vocab

    @staticmethod
    def read_data(vocab, gold_standard_file):
        with open(gold_standard_file, 'r') as f:
            full_data = [line.rstrip().split() for line in f]

        data = [x for x in full_data if all(word in vocab for word in x)]
        return data