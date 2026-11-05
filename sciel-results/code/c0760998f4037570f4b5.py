import time
import sys
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # 5 seconds margin for 90s limit

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

    # Precompute for each vertex the set of valid neighbors (for fast conflict check)
    # adj[i] already contains all j such that i+j is a square

    def is_conflict(v, current_set):
        for u in adj[v]:
            if u in current_set:
                return True
        return False

    # Greedy construction with different orderings
    def greedy_by_order(order):
        selected = set()
        for v in order:
            if not is_conflict(v, selected):
                selected.add(v)
        return selected

    # Local search: swap-based improvement
    def local_search_swap(current_set, max_iters=500):
        current_set = set(current_set)
        improved = True
        iters = 0
        while improved and iters < max_iters and (time.time() - start_time < DEADLINE):
            improved = False
            iters += 1
            # Try each vertex in the set
            for v in list(current_set):
                # Try swapping v with each vertex not in the set
                for u in range(M):
                    if u in current_set:
                        continue
                    # Check if swapping v out and u in maintains independence
                    # Remove v, check if u conflicts with remaining set
                    new_set = set(current_set)
                    new_set.remove(v)
                    if not is_conflict(u, new_set):
                        # The swap is valid (size stays same, but structure changes)
                        # We only accept if it helps future swaps, so we do it tentatively
                        # Actually for a fixed-size swap, we need a different criterion
                        # Let's use a 2-opt like move: remove v, add u, if no conflict
                        current_set = new_set
                        current_set.add(u)
                        improved = True
                        break
                if improved:
                    break
        return current_set

    # Local search: add-then-remove improvement (grow the set)
    def local_search_grow(current_set):
        current_set = set(current_set)
        improved = True
        while improved and (time.time() - start_time < DEADLINE):
            improved = False
            # Try to add any vertex that doesn't conflict
            for v in range(M):
                if v in current_set:
                    continue
                if not is_conflict(v, current_set):
                    current_set.add(v)
                    improved = True
        return current_set

    # Better local search: 1-swap that can increase size
    # Remove one vertex, then greedily add as many as possible
    def local_search_remove_and_fill(current_set):
        current_set = set(current_set)
        improved = True
        while improved and (time.time() - start_time < DEADLINE):
            improved = False
            best_new_set = None
            best_size = len(current_set)
            for v in list(current_set):
                new_set = set(current_set)
                new_set.remove(v)
                # Try to add vertices
                for u in range(M):
                    if u in new_set:
                        continue
                    if not is_conflict(u, new_set):
                        new_set.add(u)
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_new_set = set(new_set)
            if best_new_set is not None:
                current_set = best_new_set
                improved = True
        return current_set

    # Initial lower bound using multiple greedy strategies
    best_size = 0
    best_set_indices = set()

    # Strategy 1: Random restarts
    for seed in range(200):
        random.seed(seed)
        order = list(range(M))
        random.shuffle(order)
        s = greedy_by_order(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set_indices = set(s)

    # Strategy 2: Deterministic by degree (ascending)
    verts_by_degree = sorted(range(M), key=lambda x: len(adj[x]))
    s = greedy_by_order(verts_by_degree)
    if len(s) > best_size:
        best_size = len(s)
        best_set_indices = set(s)

    # Strategy 3: Deterministic by degree (descending)
    verts_by_degree_desc = sorted(range(M), key=lambda x: -len(adj[x]))
    s = greedy_by_order(verts_by_degree_desc)
    if len(s) > best_size:
        best_size = len(s)
        best_set_indices = set(s)

    # Strategy 4: Modular residue class based
    # Group by residue mod small numbers
    for mod in [3, 4, 5, 6, 7, 8, 9, 10, 12]:
        for r in range(mod):
            subset = [i for i in range(M) if vertex_map[i] % mod == r]
            if len(subset) > best_size:
                # Check if this subset is independent
                valid = True
                sset = set(subset)
                for v in subset:
                    if is_conflict(v, sset - {v}):
                        valid = False
                        break
                if valid:
                    best_size = len(subset)
                    best_set_indices = set(subset)

    # Run local search on best found
    best_set_indices = local_search_grow(best_set_indices)
    if len(best_set_indices) > best_size:
        best_size = len(best_set_indices)

    best_set_indices = local_search_remove_and_fill(best_set_indices)
    if len(best_set_indices) > best_size:
        best_size = len(best_set_indices)

    # Run local search on several random restarts
    for seed in range(50):
        random.seed(10000 + seed)
        order = list(range(M))
        random.shuffle(order)
        s = greedy_by_order(order)
        s = local_search_grow(s)
        s = local_search_remove_and_fill(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set_indices = set(s)

    # Max-Cut / partitioning heuristic
    # Assign vertices to two groups based on some criterion, then optimize
    # We try to find a large independent set by iteratively refining a partition

    def partition_heuristic(seed):
        random.seed(seed)
        # Start with random assignment to two sets
        assignment = [random.randint(0, 1) for _ in range(M)]
        
        # Iteratively improve: for each vertex, try moving it to the other set
        # if it reduces conflicts in its current set
        for iteration in range(100):
            if time.time() - start_time > DEADLINE:
                break
            changed = False
            for v in range(M):
                current_set = {u for u in range(M) if assignment[u] == assignment[v]}
                other_set = {u for u in range(M) if assignment[u] != assignment[v]}
                
                # Count conflicts in current set involving v
                conflicts_current = sum(1 for u in adj[v] if u in current_set and u != v)
                
                # If we move v to other set, count conflicts there
                conflicts_other = sum(1 for u in adj[v] if u in other_set)
                
                if conflicts_other < conflicts_current:
                    assignment[v] = 1 - assignment[v]
                    changed = True
            
            if not changed:
                break
        
        # Take the larger of the two sets that is independent
        set0 = {u for u in range(M) if assignment[u] == 0}
        set1 = {u for u in range(M) if assignment[u] == 1}
        
        # Check independence and grow
        for s in [set0, set1]:
            s = local_search_grow(s)
            s = local_search_remove_and_fill(s)
            if len(s) > best_size:
                best_size = len(s)
                best_set_indices = set(s)

    for seed in range(30):
        partition_heuristic(seed)

    # Final local search on the best
    best_set_indices = local_search_grow(best_set_indices)
    best_set_indices = local_search_remove_and_fill(best_set_indices)
    if len(best_set_indices) > best_size:
        best_size = len(best_set_indices)

    result_indices = best_set_indices
    result_values = [vertex_map[i] for i in result_indices]
    result_values.sort()

    return result_values

if __name__ == "__main__":
    res = construct()
    print(f"Found set size: {len(res)}")
