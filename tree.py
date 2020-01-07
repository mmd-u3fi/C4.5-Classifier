class Node(object):
    def __init__(self, data):
        self.dataset = data
        self.children = {}
        self.label = None
        self.is_leaf = False
    def add_child(self, edge_label, node):
        self.children[edge_label] = node
    def print_tree(self, level=0):
        print ('    ' * level + self.label + f' (level {level})')
        for child in self.children:
            self.children[child].print_tree(level+1)
