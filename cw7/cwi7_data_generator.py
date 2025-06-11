import random
import json
import csv
from collections import Counter


def generate_sample(structure: dict, data: dict):
    sample = {}
    for node in structure.keys():
        parents = structure[node]
        parent_values = tuple(sample[parent] for parent in parents)
        probabilities = data[node][str(parent_values)]
        sample[node] = random.choices([True, False], weights=probabilities)[0]
    return sample


def generate_samples(structure: dict, data: dict, n: int):
    return [generate_sample(structure, data) for _ in range(n)]


def save_samples(samples, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)


def load_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data["record_amount"], data["structure"], data["data"]


def count_truths(filename):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        truth_counts = Counter()
        for row in reader:
            for key, value in row.items():
                if value == "True":
                    truth_counts[key] += 1
    return truth_counts


def main():
    analyze_mode = False
    json_file = "cwi7_input.json"
    output_file = "cwi7_generated_data.csv"
    record_amount, structure, data = load_from_json(json_file)
    if analyze_mode:
        chair = 0
        sport = 0
        back = 0
        ache = 0
        for _ in range(50):
            samples = generate_samples(structure, data, record_amount)
            save_samples(samples, output_file)
            truth_counts = count_truths(output_file)
            chair += truth_counts["Chair"]
            sport += truth_counts["Sport"]
            back += truth_counts["Back"]
            ache += truth_counts["Ache"]
        print("Truth counts in each column:", chair / 50, sport / 50, back / 50, ache / 50)
    else:
        samples = generate_samples(structure, data, record_amount)
        save_samples(samples, output_file)


if __name__ == "__main__":
    main()
