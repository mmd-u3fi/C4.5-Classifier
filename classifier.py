from tree import Node
from dataset_info import DatasetInformation
from dataset_utils import split_on_value

class C45(object):
    def __init__(self, dataset, target_column):
        self.dataset = dataset
        self.set_info = DatasetInformation()
        self.target_column = target_column
        self.root = None
    def train(self, node):
        # BASE CASES
        # 1. if all samples belong to the same class
        if len(list(set(node.dataset['class']))) == 1:
            node.is_leaf = True
            node.label = node.dataset['class'][0]
            return
        # 2. if there are no more features to select from
        if len(node.dataset) == 1:
            node.is_leaf = True
            node.label = self.most_common_class(node.dataset, self.target_column)
            return
        # 3. if dataset is empty
        if len(node.dataset[self.target_column]) == 0:
            return
        maximum_info_column = self.find_maximum_gain(node.dataset)
        node.label = maximum_info_column
        categories = list(set(node.dataset[maximum_info_column]))
        for category in categories:
            new_set = split_on_value(node.dataset, maximum_info_column, category)
            del(new_set[maximum_info_column])
            node.add_child(category, Node(new_set))
        node.dataset = None
        for child in node.children:
            self.train(node.children[child])
    def create_tree(self):
        root = Node(self.dataset)
        self.train(root)
        self.root = root
    def find_maximum_gain(self, dataset):
        gains = {}
        maximum = None
        self.set_info.bind_dataset(dataset)
        for column in self.set_info.columns():
            if column == self.target_column:
                continue
            information_gain = self.set_info.information_gain(column, self.target_column)
            gains[column] = information_gain
            if maximum == None:
                maximum = column
            elif gains[maximum] < information_gain:
                maximum = column
        return maximum
    def most_common_class(self, dataset, column):
        data = dataset[column]
        categories = list(set(data))
        frequency = {k: 0 for k in categories}
        for category in categories:
            frequency[category] = data.count(category)
        maximum, maximum_key = 0 , None
        for key in frequency:
            if frequency[key] > maximum:
                maximum = frequency[key]
                maximum_key = key
        return maximum_key
    def evaluate(self, dataset):
        node = self.root
        tp, fp, fn, tn = 0, 0, 0, 0
        positive = 'recurrence-events'
        negative = 'no-recurrence-events'
        actual_response = None
        for i in range(len(dataset[self.target_column])):
            traveresed = []
            node = self.root
            while node.is_leaf == False:
                traveresed.append(node.children)
                print(traveresed)
                # FIXME key Error
                node = node.children[dataset[node.label][i]]
            actual_response = dataset[self.target_column][i]            
            if actual_response == positive:
                if node.label == actual_response:
                    tp += 1
                else:
                    fn += 1
            elif actual_response == negative:
                if node.label != actual_response:
                    fp += 1
                elif node.label == actual_response:
                    tn += 1
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        result = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
        result = {'recall': recall, 'accuracy': accuracy, 'precision': precision}
        return result
    def print_tree(self):
        self.root.print_tree()
    def save_tree_structure(self):
        import sys
        sys.stdout = open('tree_structure.txt', 'w+')
        self.root.print_tree()