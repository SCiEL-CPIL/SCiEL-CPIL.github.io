"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}


def construct():
    deadline = time.time() + 75.0  # safety margin before 90s kill
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)

    # Sort components by size descending - focus on largest first
    components.sort(key=len, reverse=True)

    best_path = []

    # For each component, try to find longest path using DFS with backtracking
    for comp in components:
        if time.time() > deadline:
            break
        comp_set = set(comp)
        # Only consider vertices in this component
        comp_adj = {v: [u for u in adj[v] if u in comp_set] for v in comp}
        # Sort neighbors by degree (higher degree first for exploration, or lower for leaf-finding)
        for v in comp:
            comp_adj[v].sort(key=lambda x: len(comp_adj[x]))

        best_in_comp = []

        # Try each vertex as start
        # Order starts: try lower-degree vertices first (likely endpoints)
        start_order = sorted(comp, key=lambda v: len(comp_adj[v]))

        for start in start_order:
            if time.time() > deadline:
                break
            # DFS with backtracking
            path = [start]
            path_set = {start}

            def dfs():
                nonlocal best_in_comp, path, path_set
                if time.time() > deadline:
                    return
                if len(path) > len(best_in_comp):
                    best_in_comp = list(path)
                    if len(best_in_comp) == len(comp):
                        return  # Hamiltonian path found
                current = path[-1]
                # Try extending to neighbors not yet visited
                # Sort by a heuristic: prefer neighbors with fewer unvisited neighbors (to avoid dead ends)
                candidates = [nb for nb in comp_adj[current] if nb not in path_set]
                if not candidates:
                    return
                # Heuristic ordering: prefer nodes that lead to more unvisited nodes
                # But also try to go to "leaf-like" nodes to extend paths
                # Try both orderings in different runs
                candidates.sort(key=lambda x: len([y for y in comp_adj[x] if y not in path_set]))
                for nb in candidates:
                    if time.time() > deadline:
                        return
                    path.append(nb)
                    path_set.add(nb)
                    dfs()
                    path.pop()
                    path_set.remove(nb)
                    if time.time() > deadline:
                        return
                return

            dfs()
            if time.time() > deadline:
                break

        if len(best_in_comp) > len(best_path):
            best_path = best_in_comp

    # If no path found (shouldn't happen), return a single vertex
    if not best_path:
        # Find any edge
        for u, v in P['edges']:
            return [u, v]
        return [0]

    return best_path

# EVOLVE-BLOCK-END
