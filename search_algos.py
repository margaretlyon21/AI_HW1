import time
import matplotlib.pyplot as plt
from collections import deque

# ------------------ Romania Map ------------------

romania_map = {
    'Arad': {'Zerind': 75, 'Timisoara': 118, 'Sibiu': 140},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Pitesti': 97, 'Craiova': 146},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Craiova': {'Rimnicu Vilcea': 146, 'Pitesti': 138, 'Drobeta': 120},
    'Drobeta': {'Craiova': 120, 'Mehadia': 75},
    'Mehadia': {'Drobeta': 75, 'Lugoj': 70},
    'Lugoj': {'Mehadia': 70, 'Timisoara': 111},
    'Timisoara': {'Lugoj': 111, 'Arad': 118},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101}
}

# Heuristic for BFS / A*
sld_to_bucharest = { 'Arad': 366, 'Bucharest': 0, 'Craiova': 260, 'Drobeta': 242, 'Eforie': 161, 'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151, 'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234, 'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193, 'Sibiu': 253, 'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374 }

def calculate_path_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]
    return cost


# ------------------ DFS ------------------

def depth_first_search(graph, start, goal, runs=1):
    total_time = 0
    final_path = []
    nodes_expanded = 0

    for _ in range(runs):
        stack = [(start, [start])]
        visited = set()

        start_time = time.perf_counter()

        while stack:
            current_city, path = stack.pop()
            nodes_expanded += 1

            if current_city == goal:
                final_path = path
                break

            if current_city not in visited:
                visited.add(current_city)
                for neighbor in graph[current_city]:
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))

        total_time += (time.perf_counter() - start_time)

    cost = calculate_path_cost(graph, final_path)

    print(f"\nDFS: {goal} found!")
    print(f"Path: {final_path}")
    print(f"Path length: {len(final_path)}")
    print(f"Path cost: {cost}")
    print(f"Total cities visited (nodes expanded): {nodes_expanded}")
    print(f"Time over {runs} run(s): {total_time:.8f} seconds")

    return total_time

# ------------------ BFS ------------------

def breadth_first_search(graph, start, goal, runs=1):
    total_time = 0
    final_path = []
    nodes_expanded = 0

    for _ in range(runs):
        queue = deque([(start, [start])])
        visited = set()

        start_time = time.perf_counter()

        while queue:
            current_city, path = queue.popleft()
            nodes_expanded += 1

            if current_city == goal:
                final_path = path
                break

            if current_city not in visited:
                visited.add(current_city)
                for neighbor in graph[current_city]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

        total_time += (time.perf_counter() - start_time)

    cost = calculate_path_cost(graph, final_path)

    print(f"\nBFS: {goal} found!")
    print(f"Path: {final_path}")
    print(f"Path length: {len(final_path)}")
    print(f"Path cost: {cost}")
    print(f"Total cities visited (nodes expanded): {nodes_expanded}")
    print(f"Time over {runs} run(s): {total_time:.8f} seconds")

    return total_time


# ------------------ Informed Searches ------------------
def best_first_search(graph, start, goal, heuristic):
    return

def a_star_search(graph, start, goal, heuristic):
    return

# ------------------ Main ------------------

if __name__ == "__main__":
    depth_first_search(romania_map, 'Arad', 'Bucharest', runs=10000)
    breadth_first_search(romania_map, 'Arad', 'Bucharest', runs=10000)

    best_first_search(
        graph=romania_map,
        start='Arad',
        goal='Bucharest',
        heuristic=sld_to_bucharest
    )
    a_star_search(
        romania_map,
        start='Arad',
        goal='Bucharest',
        heuristic=sld_to_bucharest
    )
