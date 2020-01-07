def parse_dataset():
    dataset = {
        'age': [],
        'menopause': [],
        'tumor-size': [],
        'inv-nodes': [],
        'node-caps': [],
        'deg-malig': [],
        'breast': [],
        'breast-quad': [],
        'irradiant': [],
        'class': [],
    }

    with open("Breast_Cancer_dataset.txt", 'r') as f:
        for line in f.read().split('\n'):
            if '?' in line:
                continue
            line = line.split(',')
            for index, feature in enumerate(line):
                feature = feature.strip("'")
                if index == 0:
                    dataset['age'].append(feature)
                elif index == 1:
                    dataset['menopause'].append(feature)
                elif index == 2:
                    dataset['tumor-size'].append(feature)
                elif index == 3:
                    dataset['inv-nodes'].append(feature)
                elif index == 4:
                    dataset['node-caps'].append(feature)
                elif index == 5:
                    dataset['deg-malig'].append(feature)
                elif index == 6:
                    dataset['breast'].append(feature)
                elif index == 7:
                    dataset['breast-quad'].append(feature)
                elif index == 8:
                    dataset['irradiant'].append(feature)
                elif index == 9:
                    dataset['class'].append(feature)
    return dataset