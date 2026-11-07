"""Improved program for cochromatic partition of pinned random graph."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    es = set()
    for e in edges:
        es.add((e[0], e[1]))
        es.add((e[1], e[0]))

    # Precompute adjacency as bitmasks for speed
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def is_clique(vertices):
        """Check if vertices form a clique."""
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if (1 << v) & adj[u] == 0:
                    return False
        return True

    def is_independent(vertices):
        """Check if vertices form an independent set."""
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if (1 << v) & adj[u] != 0:
                    return False
        return True

    # Find maximum clique and maximum independent set
    max_clique = []
    max_indep = []

    # Try all subsets in decreasing size to find max clique
    for size in range(n, 0, -1):
        found = False
        for combo in combinations(range(n), size):
            if is_clique(list(combo)):
                max_clique = list(combo)
                found = True
                break
        if found:
            break

    # Find max independent set (clique in complement)
    for size in range(n, 0, -1):
        found = False
        for combo in combinations(range(n), size):
            if is_independent(list(combo)):
                max_indep = list(combo)
                found = True
                break
        if found:
            break

    # Build initial greedy solution
    def greedy_partition():
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                if is_clique(trial) or is_independent(trial):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    best_parts = greedy_partition()
    best_count = len(best_parts)

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 60s limit

    # Local search: try to merge parts or move vertices to reduce count
    def try_improve(parts):
        """Try to improve the partition by moving vertices or merging parts."""
        improved = True
        while improved:
            improved = False
            parts = [list(p) for p in parts]

            # Try to move each vertex to another part
            for i in range(len(parts)):
                for v in list(parts[i]):
                    if len(parts[i]) == 1:
                        continue
                    # Try moving v to another part
                    for j in range(len(parts)):
                        if i == j:
                            continue
                        trial = parts[j] + [v]
                        if is_clique(trial) or is_independent(trial):
                            parts[i].remove(v)
                            parts[j].append(v)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # Try to merge two parts
            for i in range(len(parts)):
                for j in range(i+1, len(parts)):
                    merged = parts[i] + parts[j]
                    if is_clique(merged) or is_independent(merged):
                        parts[i] = merged
                        parts.pop(j)
                        improved = True
                        break
                if improved:
                    break

            # Remove empty parts
            parts = [p for p in parts if p]

        return parts

    best_parts = try_improve(best_parts)
    best_count = len(best_parts)

    # More aggressive search: try all possible assignments of small subsets
    # to reduce the number of parts
    if time.time() < deadline:
        # Try swapping vertices between parts
        for _ in range(100):
            if time.time() > deadline:
                break
            if len(best_parts) <= 2:
                break
            # Randomly pick two parts and try to redistribute
            import random
            random.seed(42)  # Deterministic
            parts = [list(p) for p in best_parts]
            i, j = random.sample(range(len(parts)), 2)
            # Try moving one vertex from larger to smaller
            if len(parts[i]) < len(parts[j]):
                i, j = j, i
            for v in list(parts[i]):
                trial = parts[j] + [v]
                if is_clique(trial) or is_independent(trial):
                    parts[i].remove(v)
                    parts[j].append(v)
                    new_parts = [p for p in parts if p]
                    if len(new_parts) < len(best_parts):
                        best_parts = new_parts
                        best_count = len(best_parts)
                        break
            # Try merging
            for i in range(len(best_parts)):
                for j in range(i+1, len(best_parts)):
                    merged = best_parts[i] + best_parts[j]
                    if is_clique(merged) or is_independent(merged):
                        new_parts = list(best_parts[:i]) + [merged] + list(best_parts[j+1:])
                        if len(new_parts) < len(best_parts):
                            best_parts = new_parts
                            best_count = len(best_parts)

    # Convert to proper Python types
    result = []
    for part in best_parts:
        result.append([int(v) for v in part])

    return result
# EVOLVE-BLOCK-END
