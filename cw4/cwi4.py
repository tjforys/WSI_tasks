from ucimlrepo import fetch_ucirepo 
import pandas as pd
import math
import collections


def calculate_entropy(data):
    entropy = 0
    target = data[data.columns[-1]]
    dict = collections.Counter(target.values)
    total = sum(dict.values())
    for value in dict.values():
        entropy -= (value / total) * math.log(value / total, 2)
    return entropy


def calculate_info_gain(data, feature, curr_entropy):
    unique_values = data[feature].unique()
    weighted_entropy = 0
    for value in unique_values:
        subset = data[data[feature] == value]
        proportion = len(subset) / len(data)
        weighted_entropy += proportion * calculate_entropy(subset)

    information_gain = curr_entropy - weighted_entropy

    return information_gain


def choose_best_feat(data, curr_entropy):
    max_gain = -math.inf

    for column in data.columns[:-1]:
        information_gain = calculate_info_gain(data, column, curr_entropy)
        if information_gain > max_gain:
            max_gain = information_gain
            max_feat = column

    return max_feat


def all_same_class(data):
    target = data[data.columns[-1]]
    target = target.to_numpy()
    return (target[0] == target).all()


def build_tree(data):
    if data.empty:
        return ('Leaf', None)
    if all_same_class(data):
        return ('Leaf', data[data.columns[-1]].values[0])
    if len(data.columns) == 1:
        return ('Leaf', collections.Counter(data[data.columns[-1]]).most_common(1)[0][0])
    curr_entropy = calculate_entropy(data)
    best_feat = choose_best_feat(data, curr_entropy)
    tree = ('Node', best_feat, {})

    feat_values = [str(value) for value in data[best_feat].unique()]

    subsets = {}

    for value in feat_values:
        subsets[value] = data[data[best_feat] == value].drop(best_feat, axis=1)
    for val, subset in subsets.items():
        subtree = build_tree(subset)
        tree[2][val] = subtree
    return tree


def predict(tree, row):
    if tree[0] == 'Leaf':
        return tree[1]
    else:
        attr = tree[1]
        branches = tree[2]
        val = str(row[attr])
        if val in branches:
            return predict(branches[val], row)
        else:
            return None


def evaluate(tree, data):
    good = 0
    bad = 0
    for index, row in data.iterrows():
        expected = row.iloc[-1]
        predicted = predict(tree, row)
        # print(expected, predicted)
        if expected == predicted:
            good += 1
        else:
            bad += 1
    return good, bad

# dataset = fetch_ucirepo(id=14) 
dataset = fetch_ucirepo(id=73) 

dataset2 = dataset.data.original
dataset2 = dataset2.astype(str)

dataset2 = dataset2.iloc[:, [2,3,4,5,6,7,8,22]]
print(dataset2)

minprec = math.inf
maxprec = -math.inf
good = 0
bad = 0


for i in range(50):
    training = dataset2.sample(frac=0.6)
    testing = dataset2.drop(training.index)

    tree = build_tree(training)
    tempgood, tempbad = evaluate(tree, testing)
    precision = tempgood / (tempgood + tempbad)
    minprec = min(minprec, precision)
    maxprec = max(maxprec, precision)
    good += tempgood
    bad += tempbad
print(minprec, maxprec, good, bad, good/(good+bad))