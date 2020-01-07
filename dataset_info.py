from math import log2

class DatasetInformation(object):
    def __init__(self):
        self.dataset = None
    def bind_dataset(self, dataset):
        self.dataset = dataset
    def columns(self):
        return self.dataset.keys()
    def class_frequency(self, column):
        frequencies = {}
        for value in self.dataset[column]:
            if value in frequencies:
                frequencies[value] += 1
            else:
                frequencies[value] = 1
        return frequencies
    def sample_size(self, column):
        return len(self.dataset[column])
    def entropy(self, column):
        frequencies = self.class_frequency(column)
        set_size = self.sample_size(column)
        H = 0
        for label in frequencies:
            p = frequencies[label] / set_size
            H -= p * log2(p)
        return H
    def information_gain(self, posterior_column, target_column):
        frequencies = self.class_frequency(posterior_column)
        probablities = {k:{} for k in frequencies}
        for condition, result in zip(self.dataset[posterior_column], self.dataset[target_column]):
            if result in probablities[condition]:
                probablities[condition][result] += 1
            else:
                probablities[condition][result] = 1
        gain = 0
        for class_label in probablities:
            conditional_entropy = 0
            for result_label in probablities[class_label]:
                probablities[class_label][result_label] /= frequencies[class_label]
                p = probablities[class_label][result_label]
                conditional_entropy -= p * log2(p)
            weight = frequencies[class_label] / self.sample_size(target_column)
            conditional_entropy *= weight
            gain += conditional_entropy
        gain = self.entropy(target_column) - gain
        split_info = self.entropy(posterior_column)
        if split_info == 0:
            return 0
        return gain / split_info