import heapq
import time
from collections import deque


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

sld_to_bucharest = {
    'Arad': 366,
    'Bucharest': 0,
    'Craiova': 260,
    'Drobeta': 242,
    'Eforie': 161,
    'Fagaras': 176,
    'Giurgiu': 77,
    'Hirsova': 151,
    'Iasi': 226,
    'Lugoj': 244,
    'Mehadia': 241,
    'Neamt': 234,
    'Oradea': 380,
    'Pitesti': 100,
    'Rimnicu Vilcea': 193,
    'Sibiu': 253,
    'Timisoara': 329,
    'Urziceni': 80,
    'Vaslui': 199,
    'Zerind': 374
}

def depth_first_search(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    nodes_expanded = 0

    while stack:
        current_city, path = stack.pop()
        nodes_expanded += 1

        if current_city == goal:
            print("\nDFS: Goal found!")
            print(f"Path: {path}")
            print(f"Total cities visited (nodes expanded): {nodes_expanded}")
            return

        if current_city not in visited:
            visited.add(current_city)

            for neighbor in graph[current_city]:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    print("\nDFS: No path found.")
    print(f"Total cities visited (nodes expanded): {nodes_expanded}")
    return

def breadth_first_search(graph, start, goal):
    queue = deque([(start, [start])])
    visited = set()
    nodes_expanded = 0

    while queue:
        current_city, path = queue.popleft()
        nodes_expanded += 1

        if current_city == goal:
            print("\nBFS: Goal found!")
            print(f"Path: {path}")
            print(f"Total cities visited (nodes expanded): {nodes_expanded}")
            return

        if current_city not in visited:
            visited.add(current_city)

            for neighbor in graph[current_city]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    print("\nBFS: No path found.")
    print(f"Total cities visited (nodes expanded): {nodes_expanded}")
    return


def best_first_search(graph, start, goal, heuristic):
    return

def a_star_search(graph, start, goal, heuristic):
    return

if __name__ == "__main__":
    depth_first_search(romania_map, 'Arad', 'Bucharest')
    breadth_first_search(romania_map, 'Arad', 'Bucharest')
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


