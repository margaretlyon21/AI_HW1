import time
import matplotlib.pyplot as plt
from collections import deque
import heapq

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

def heuristic_1(graph, city, goal, sld_to_bucharest):
    return abs(sld_to_bucharest[city] - sld_to_bucharest[goal])
    # return abs(sld_to_bucharest.get(city, 0) - sld_to_bucharest.get(goal, 0))

def heuristic_2(graph, city, goal, sld_to_bucharest):
    distances = []
    to_bucharest = sld_to_bucharest[goal]

    neighbors = graph.get(city, {})
    if not neighbors:
        return heuristic_1(graph, city, goal, sld_to_bucharest)

    for neighbor, distance in neighbors.items():
        distances.append(distance + abs(sld_to_bucharest[neighbor] - to_bucharest))
    return min(distances)
    
# ------------------ Greedy Algorithm ------------------
def greedy(graph, start, goal, heuristic_fn, number_of_iterations):
    total_time = 0

    for _ in range(number_of_iterations):
        # re-init per iteration so each run is independent
        frontier = [(heuristic_fn(romania_map, start, goal, sld), start)]
        came_from = {}
        cost_so_far = {start: 0}
        cities_visited = 0
        estimated_distance = 0

        # start timing the actual work for this iteration
        start_time = time.perf_counter()

        while frontier:
            _, current = heapq.heappop(frontier)
            cities_visited += 1

            #sucessfully reached city
            if current == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                    estimated_distance += dict(romania_map[path[-2]])[path[-1]]
                path.reverse()
                # city is found, so it must end the loop
                # return cities_visited
                break

            #visit neighbors
            for next_node, distance in graph[current]:
                new_cost = cost_so_far[current] + distance
                #if node is not visited or if node's cost is cheaper
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = heuristic_fn(romania_map, next_node, goal, sld)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current

        total_time += (time.perf_counter() - start_time)

    print(f"\nBFS: {goal} found!")
    print(f"Path: {path}")
    print(f"Path length: {len(path)}")
    print(f"Path cost: {estimated_distance}")
    print(f"Total cities visited (nodes expanded): {cities_visited}")
    print(f"Time over {number_of_iterations} run(s): {total_time:.8f} seconds")

# ------------------ A* Algorithm ------------------

def A_algorithm(graph, start, goal, runs=1, heuristic_choice=1):
    total_time = 0
    final_path = []
    nodes_expanded = 0

    if heuristic_choice == 1:
        def Heuristic(city): return heuristic_1(graph, city, goal, sld_to_bucharest)
    elif heuristic_choice == 2:
        def Heuristic(city): return heuristic_2(graph, city, goal, sld_to_bucharest)

    for _ in range(runs):
        pqueue = []
        tiebreaker = 0
        
        start_time = time.perf_counter()
        
        start_H = Heuristic(start)
        heapq.heappush(pqueue, (start_H, tiebreaker, start, 0, [start]))
        
        best_cost = {start: 0}

        while pqueue:
            #pop the evaluation value, and ignore the tiebreaker 
            popped_eval_val, _, city, current_cost, path = heapq.heappop(pqueue)
            nodes_expanded += 1

            if city == goal:
                final_path = path
                break

            for neighbor, neighbor_cost in graph.get(city, {}).items():
                trial_cost = current_cost + neighbor_cost
                if trial_cost < best_cost.get(neighbor, float('inf')):
                    best_cost[neighbor] = trial_cost
                    tiebreaker += 1
                    new_eval_val = trial_cost + Heuristic(neighbor)
                    heapq.heappush(pqueue, (new_eval_val, tiebreaker, neighbor, trial_cost, path + [neighbor]))

        total_time += (time.perf_counter() - start_time)

    total_cost = calculate_path_cost(graph, final_path)

    if heuristic_choice == 1:
        print(f"\nA* using Heuristic 1: {goal} found!")
    elif heuristic_choice == 2:
        print(f"\nA* using Heuristic 2: {goal} found!")
    print(f"Path: {final_path}")
    print(f"Path length: {len(final_path)}")
    print(f"Path cost: {total_cost}")
    print(f"Total cities visited (nodes expanded): {nodes_expanded}")
    print(f"Time over {runs} run(s): {total_time:.8f} seconds")

    return total_time
    

# ------------------ Main ------------------

if __name__ == "__main__":
    depth_first_search(romania_map, 'Arad', 'Bucharest', runs=10000)
    breadth_first_search(romania_map, 'Arad', 'Bucharest', runs=10000)

    greedy(romania_map, 'Arad', 'Bucharest', heuristics_1)
    
    A_algorithm(
        graph=romania_map,
        start='Arad',
        goal='Bucharest',
        runs=10000,
        heuristic_choice=1
        )
    
    A_algorithm(
        graph=romania_map,
        start='Arad',
        goal='Bucharest',
        runs=10000,
        heuristic_choice=2
        )

