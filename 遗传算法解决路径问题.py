import os
import numpy as np
import random
import matplotlib.pyplot as plt


import pandas as pd
output_path = "E:/结果/"
if not os.path.exists(output_path):
    os.makedirs(output_path)


# 设置 Matplotlib 的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

cities = {
    1: (33.5, 29.3), 2: (33.0, 69.6), 3: (9.0, 71.3), 4: (69.0, 76.1), 5: (13.3, 67.5),
    6: (7.7, 87.4), 7: (64.8, 61.0), 8: (50.6, 22.2), 9: (78.5, 83.4), 10: (24.1, 29.9),
    11: (33.9, 0.5), 12: (66.7, 26.3), 13: (91.6, 10.6), 14: (71.0, 52.4), 15: (12.4, 16.8),
    16: (22.1, 33.9), 17: (55.9, 25.4), 18: (7.0, 80.4), 19: (63.0, 91.5), 20: (34.7, 98.4)
}



def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def create_distance_matrix(cities):
    n = len(cities)
    distance_matrix = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                distance_matrix[i - 1][j - 1] = calculate_distance(cities[i], cities[j])
            else:
                distance_matrix[i - 1][j - 1] = float('inf')  # 同一城市距离设置为无穷大
    return distance_matrix



def initialize_population(num_individuals, num_cities):
    population = []
    for _ in range(num_individuals):
        individual = np.random.permutation(num_cities) + 1
        population.append(individual)
    return population



def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i] - 1][route[i + 1] - 1]
    total_distance += distance_matrix[route[-1] - 1][route[0] - 1]  # 回到起点的距离
    return total_distance



def selection(population, scores, num_parents):
    scores_inv = 1 / np.array(scores)
    probabilities = scores_inv / np.sum(scores_inv)
    parents_indices = np.random.choice(len(population), size=num_parents, replace=True, p=probabilities)
    parents = [population[idx] for idx in parents_indices]
    return parents



def crossover(parent1, parent2):
    size = len(parent1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    new_p1 = [None] * size
    new_p2 = [None] * size
    new_p1[cxpoint1:cxpoint2 + 1] = parent1[cxpoint1:cxpoint2 + 1]
    new_p2[cxpoint1:cxpoint2 + 1] = parent2[cxpoint1:cxpoint2 + 1]

    def fill_remaining(new_child, parent):
        pos = (cxpoint2 + 1) % size
        for city in parent:
            if city not in new_child:
                new_child[pos] = city
                pos = (pos + 1) % size

    fill_remaining(new_p1, parent2)
    fill_remaining(new_p2, parent1)
    return new_p1, new_p2

def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        swap_idx1, swap_idx2 = random.sample(range(len(route)), 2)
        route[swap_idx1], route[swap_idx2] = route[swap_idx2], route[swap_idx1]
    return route


def save_route_plot(route, cities, iteration):
    x_coords = [cities[city][0] for city in route]
    y_coords = [cities[city][1] for city in route]
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, 'bo-', label='path')
    plt.scatter(x_coords, y_coords, color='red')

    for city, (x, y) in zip(route, zip(x_coords, y_coords)):
        plt.text(x, y, str(city), color='black', fontsize=12)

    plt.title(f' - Run {iteration}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks(np.arange(0, 100, 5))
    plt.yticks(np.arange(0, 100, 5))
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_path}route_plot_run_{iteration}.png')
    plt.close()


# 运行10次
results = []
num_individuals = 5000
num_generations = 50000
mutation_rate = 0.1
distance_matrix = create_distance_matrix(cities)

for i in range(1, 11):
    best_distance = float('inf')
    best_route = None
    population = initialize_population(num_individuals, len(cities))



    improvement_threshold = 0.01  # 显著改进的阈值
    max_no_improve_generations = 5000  # 连续无显著改进的最大代数
    generations_without_improvement = 0  # 计数无显著改进的代数

    for generation in range(num_generations):
        scores = [calculate_total_distance(ind, distance_matrix) for ind in population]
        current_best_distance = min(scores)

        # 检查是否有显著改进
        if current_best_distance < best_distance - improvement_threshold:
            best_distance = current_best_distance
            best_route = population[scores.index(best_distance)]
            generations_without_improvement = 0  # 重置计数器
        else:
            generations_without_improvement += 1

        # 检查无显著改进的最大代数是否达到上限
        if generations_without_improvement >= max_no_improve_generations:
            print(f"Stopped early after {generation + 1} generations due to no significant improvement.")
            break

        parents = selection(population, scores, num_individuals // 2)
        next_generation = []
        for j in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[j], parents[j + 1]
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            next_generation.extend([offspring1, offspring2])
        population = next_generation

    results.append({
        "运行次数": i,
        "最优路径": best_route,
        "最短距离": best_distance
    })

    save_route_plot(best_route, cities, i)

results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_path}循环10次的结果.xlsx", index=False, encoding='utf-8-sig')
