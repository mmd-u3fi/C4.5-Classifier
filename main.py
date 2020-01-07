from dataset_parser import parse_dataset
from dataset_info import DatasetInformation
from classifier import C45
from dataset_utils import split_on_value
from test_train_split import dataset_split

dataset = parse_dataset()
train, test = dataset_split(dataset, 0.25)

d_tree = C45(train, 'class')
d_tree.create_tree()
print(d_tree.evaluate(test))