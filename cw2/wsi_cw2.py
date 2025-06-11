from cec2017.functions import f2, f13
import numpy as np


def mutate(points, sigma, cube_bound):
    points2 = []
    for point in points:
        newpoint = np.random.normal(point, sigma)
        newpoint = np.array([max(-cube_bound, min(cube_bound, num)) for num in newpoint])
        points2.append(newpoint)
    return points2


def tournament_selection(points, point_values, population):
    new_points = []
    for i in range(population):
        first_point_index = np.random.randint(0, population)
        second_point_index = np.random.randint(0, population)
        if point_values[first_point_index] <= point_values[second_point_index]:
            new_points.append(points[first_point_index])
        else:
            new_points.append(points[second_point_index])

    return new_points


def optimalization(points, test_func, iterations):
    i = 0
    while i < iterations:

        point_values = [test_func(x) for x in points]
        points = tournament_selection(points, point_values, population)
        points = mutate(points, sigma, cube_bound)
        i += 1

    return min(point_values)


def evaluate(run_number, population, cube_bound, calculation_budget, dimensionality, func):
    result_list = []
    for i in range(run_number):
        points = []
        for i in range(population):
            points.append(np.random.uniform(-cube_bound, cube_bound, size=dimensionality))
        result_list.append(optimalization(points, func, iterations=calculation_budget//population))

    print("f2:")
    print("avg: ", '%.4E' % (sum(result_list)/len(result_list)))
    print("min: ", '%.4E' % min(result_list))
    print("max: ", '%.4E' % max(result_list))
    print("std dev: ", '%.4E' % np.std(result_list))


if __name__ == "__main__":
    calculation_budget = 10000
    cube_bound = 100
    dimensionality = 10
    population = 32
    sigma = 1
    run_number = 50

    iterations = calculation_budget // population

    print("population: ", population)
    print("mutation strength: ", sigma)

    evaluate(run_number, population, cube_bound, calculation_budget, dimensionality, f2)
    evaluate(run_number, population, cube_bound, calculation_budget, dimensionality, f13)
