import time
import random
import sys
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # 5 seconds margin for 60s limit

    N = 256
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Identify vertices that cannot be in the set (2*v is square)
    forbidden = set()
    for v in range(1, N + 1):
        if 2 * v in squares:
            forbidden.add(v)

    valid_vertices = [v for v in range(1, N + 1) if v not in forbidden]
    M = len(valid_vertices)
    vertex_map = valid_vertices  # i -> v

    # Build adjacency list for valid vertices
    adj = [set() for _ in range(M)]
    for i in range(M):
        vi = vertex_map[i]
        for j in range(i + 1, M):
            vj = vertex_map[j]
            if vi + vj in squares:
                adj[i].add(j)
                adj[j].add(i)

    # Greedy independent set with random restarts
    best_size = 0
    best_set_indices = set()

    def greedy_random(seed):
        random.seed(seed)
        order = list(range(M))
        random.shuffle(order)
        selected = set()
        for v in order:
            conflict = any(u in selected for u in adj[v])
            if not conflict:
                selected.add(v)
        return selected

    # Deterministic greedy by degree
    def greedy_deterministic():
        verts = sorted(range(M), key=lambda x: len(adj[x]))
        selected = set()
        for v in verts:
            if v in selected:
                continue
            conflict = any(u in selected for u in adj[v])
            if not conflict:
                selected.add(v)
        return selected

    # Run multiple strategies to get a strong initial lower bound
    for i in range(100):
        s = greedy_random(i)
        if len(s) > best_size:
            best_size = len(s)
            best_set_indices = set(s)

    s = greedy_deterministic()
    if len(s) > best_size:
        best_size = len(s)
        best_set_indices = set(s)

    # Local search: try to add vertices one by one, then try swaps
    def local_search(initial_set):
        selected = set(initial_set)
        improved = True
        while improved:
            improved = False
            # Try adding each unselected vertex
            for v in range(M):
                if v in selected:
                    continue
                if not any(u in selected for u in adj[v]):
                    selected.add(v)
                    improved = True
            # Try swapping: remove one selected, add one or more unselected
            for v in list(selected):
                without_v = selected - {v}
                # Try adding vertices that conflict with v but not with without_v
                candidates = [u for u in range(M) if u not in without_v and not any(w in without_v for w in adj[u])]
                for u in candidates:
                    new_size = len(without_v) + 1
                    if new_size > len(selected):
                        selected = without_v | {u}
                        improved = True
                        break
        return selected

    # Apply local search to the best found so far
    improved_set = local_search(best_set_indices)
    if len(improved_set) > best_size:
        best_size = len(improved_set)
        best_set_indices = improved_set

    # Branch and bound with tighter bounds
    sys.setrecursionlimit(10000)

    def greedy_matching_size(verts):
        """Compute a greedy matching size for upper bound."""
        matched = set()
        match_size = 0
        for v in verts:
            if v in matched:
                continue
            for u in adj[v]:
                if u in verts and u not in matched:
                    matched.add(v)
                    matched.add(u)
                    match_size += 1
                    break
        return match_size

    def solve(remaining, current_size, current_set):
        nonlocal best_size, best_set_indices

        if time.time() - start_time > DEADLINE:
            return

        if not remaining:
            if current_size > best_size:
                best_size = current_size
                best_set_indices = set(current_set)
            return

        # Upper bound
        gm = greedy_matching_size(remaining)
        ub = len(remaining) - gm
        if current_size + ub <= best_size:
            return

        # Pick vertex with highest degree in remaining
        max_deg = -1
        v_choice = None
        for v in remaining:
            deg = sum(1 for u in adj[v] if u in remaining)
            if deg > max_deg:
                max_deg = deg
                v_choice = v
                if deg == len(remaining) - 1:
                    break

        if v_choice is None:
            return

        # Branch 1: Include v_choice
        new_remaining = set(remaining)
        new_remaining.remove(v_choice)
        for u in adj[v_choice]:
            if u in new_remaining:
                new_remaining.remove(u)
        new_set = set(current_set)
        new_set.add(v_choice)
        solve(new_remaining, current_size + 1, new_set)

        if time.time() - start_time > DEADLINE:
            return

        # Branch 2: Exclude v_choice
        new_remaining2 = set(remaining)
        new_remaining2.remove(v_choice)
        solve(new_remaining2, current_size, current_set)

    try:
        solve(set(range(M)), 0, set())
    except RecursionError:
        pass

    # Final local search on the best found
    improved_set = local_search(best_set_indices)
    if len(improved_set) > best_size:
        best_size = len(improved_set)
        best_set_indices = improved_set

    result_indices = best_set_indices
    result_values = [vertex_map[i] for i in result_indices]
    result_values.sort()

    return result_values

if __name__ == "__main__":
    res = construct()
    print(f"Found set size: {len(res)}")
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    valid = True
    for a in res:
        if 2 * a in squares:
            valid = False
            print(f"Invalid diagonal: {a}")
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            if res[i] + res[j] in squares:
                valid = False
                print(f"Invalid pair: {res[i]}, {res[j]}")
    if valid:
        print("Valid")
    else:
        print("Invalid")
