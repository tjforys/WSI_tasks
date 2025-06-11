import itertools
import time
import random

m = []
p = []
item_number = 10

for i in range(item_number):
    m.append(random.randint(10, 20))
    p.append(random.randint(1, 20))


Mass = sum(m)/2
  
items = [[m[i], p[i]] for i in range(len(m))]


print("Initial items: ", items)

#brute_force approach
solution_list = [list(x) for i in range(1, len(items)+1) for x in itertools.combinations(items, i) ]

start_time = time.time()
max_profit = 0
max_solution = []

for solution in solution_list:
    if sum(a[0] for a in solution) > Mass:
        continue
    if sum(a[1] for a in solution) > max_profit:
        max_solution = solution
        max_profit = sum(a[1] for a in solution)

end_time = time.time() - start_time

print("Bruteforce approach: ")
print("time: ", end_time)
print("profit: ", max_profit)
print("solution: ", max_solution)

#heurystic approach
start_time = time.time()
items.sort(reverse=True, key=lambda x: x[1]/x[0])
mass_left = Mass
min_mass = sorted(items, key=lambda x: x[0])[0][0]
max_solution = []
value = 0
for item in items:
    if mass_left < min_mass:
        break
    if mass_left > item[0]:
        mass_left -= item[0]
        value += item[1]
        max_solution.append(item)

end_time = time.time() - start_time

print("Heurystic approach: ")
print("time: ", end_time)
print("profit: ", max_profit)
print("solution: ", max_solution)
