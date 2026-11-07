"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degrees_in_subset(subset, adj):
    """Compute degree of each vertex in subset within the induced subgraph."""
    subset_set = set(subset)
    degrees = {}
    for v in subset:
        deg = sum(1 for u in subset if u != v and u in adj[v])
        degrees[v] = deg
    return degrees

def _score_subset(subset, adj):
    """Return number of distinct degrees in induced subgraph."""
    if len(subset) < P['minimum_vertices']:
        return -1
    degrees = _degrees_in_subset(subset, adj)
    return len(set(degrees.values()))

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    n = P['n']
    adj = _build_adj()
    
    best_score = 0
    best_subset = list(range(8))  # fallback
    
    # Initial evaluation
    init_score = _score_subset(list(range(9)), adj)
    if init_score > best_score:
        best_score = init_score
        best_subset = list(range(9))
    
    # Strategy 1: Try all subsets of size 8, 9, 10 (combinatorial for small sizes)
    # C(17,8) = 24310, C(17,9) = 24310, C(17,10) = 19448 - all feasible
    for size in range(8, 13):
        if time.time() > deadline:
            break
        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
            score = _score_subset(subset, adj)
            if score > best_score:
                best_score = score
                best_subset = list(subset)
                # If we found all possible distinct degrees (size), we can't do better
                if best_score == size:
                    return best_subset
    
    # Strategy 2: Local search from best subset
    if time.time() < deadline:
        current = list(best_subset)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding a vertex
            for v in range(n):
                if v in current:
                    continue
                new_subset = current + [v]
                score = _score_subset(new_subset, adj)
                if score > current_score:
                    current = new_subset
                    current_score = score
                    best_score = score
                    best_subset = list(current)
                    improved = True
                    if current_score == len(current):
                        return best_subset
            # Try removing a vertex (if size > 8)
            if len(current) > 8:
                for v in list(current):
                    new_subset = [x for x in current if x != v]
                    score = _score_subset(new_subset, adj)
                    if score > current_score:
                        current = new_subset
                        current_score = score
                        best_score = score
                        best_subset = list(current)
                        improved = True
            # Try swapping
            if time.time() < deadline:
                for v_in in list(current):
                    for v_out in range(n):
                        if v_out in current:
                            continue
                        new_subset = [x for x in current if x != v_in] + [v_out]
                        score = _score_subset(new_subset, adj)
                        if score > current_score:
                            current = new_subset
                            current_score = score
                            best_score = score
                            best_subset = list(current)
                            improved = True
                            if current_score == len(current):
                                return best_subset
    
    # Ensure we return at least 8 vertices
    if len(best_subset) < 8:
        best_subset = list(range(8))
    
    # Convert to ensure Python int types
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
