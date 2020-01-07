def dataset_split(dataset, ratio):
    size = len(dataset[list(dataset.keys())[0]])
    test_size = int(size * ratio)
    test_dataset = {k: [v for v in dataset[k][:test_size]] for k in dataset}
    train_dataset = {k: [v for v in dataset[k][test_size:size]] for k in dataset}
    return (train_dataset, test_dataset)
    