def split_on_value(dataset, column, value):
    new_dataset = {k: [v for ind, v in enumerate(dataset[k]) if dataset[column][ind] == value] \
        for k in dataset}
    return new_dataset