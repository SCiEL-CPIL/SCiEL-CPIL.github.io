"""Constraint-based backtracking solver for snake-in-the-box in Q_9."""
import time
import random
import numpy as np

N = 9
SIZE = 1 << N
DEADLINE = 145.0  # safety margin before 150s kill

# Precompute neighbors
_NEIGHBORS = None
def _get_neighbors():
    global _NEIGHBORS
    if _NEIGHBORS is None:
        _NEIGHBORS = np.zeros((SIZE, N), dtype=np.int32)
        for v in range(SIZE):
            for b in range(N):
                _NEIGHBORS[v, b] = v ^ (1 << b)
    return _NEIGHBORS

def _greedy_seed():
    """Generate a greedy snake as a starting baseline."""
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v is adjacent to any vertex in path[:-1]
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    return path

def _dfs_solve(path, used_mask, deadline, best_path_ref):
    """
    Depth-first search with forward checking and MRV heuristic.
    path: list of vertices in order
    used_mask: numpy bool array marking used vertices
    deadline: time limit
    best_path_ref: list that will hold the best path found
    """
    neighbors = _get_neighbors()
    
    if len(path) > len(best_path_ref[0]):
        best_path_ref[0] = path[:]
    
    if time.time() > deadline:
        return
    
    cur = path[-1]
    path_len = len(path)
    
    # Collect all valid extensions
    candidates = []
    for b in range(N):
        cand = int(neighbors[cur, b])
        if used_mask[cand]:
            continue
        # Check adjacency to all previous vertices except the last
        valid = True
        for i in range(path_len - 1):
            if (cand ^ path[i]).bit_count() == 1:
                valid = False
                break
        if valid:
            candidates.append(cand)
    
    if not candidates:
        return
    
    # MRV heuristic: order candidates by how many extensions they have
    # (fewer extensions first, to prune early)
    def _count_extensions(cand, used_mask, path, path_len):
        count = 0
        for b in range(N):
            next_cand = cand ^ (1 << b)
            if used_mask[next_cand]:
                continue
            # Check if next_cand is adjacent to any vertex in path
            valid = True
            for i in range(path_len):
                if (next_cand ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if valid:
                count += 1
        return count
    
    # For speed, only compute extension counts if we have many candidates
    if len(candidates) > 1:
        scored = [(c, _count_extensions(c, used_mask, path, path_len)) for c in candidates]
        scored.sort(key=lambda x: x[1])  # Fewest extensions first (MRV)
        candidates = [c for c, _ in scored]
    
    # Try each candidate
    for cand in candidates:
        if time.time() > deadline:
            return
        
        path.append(cand)
        used_mask[cand] = True
        
        _dfs_solve(path, used_mask, deadline, best_path_ref)
        
        path.pop()
        used_mask[cand] = False

def _iterative_deepening_dfs(start_path, deadline, best_path_ref):
    """
    Iterative deepening DFS: try to extend the path by at least k more vertices.
    """
    neighbors = _get_neighbors()
    used_mask = np.zeros(SIZE, dtype=bool)
    for v in start_path:
        used_mask[v] = True
    
    # Try different target lengths
    base_len = len(start_path)
    for target_extra in [1, 2, 3, 5, 8, 12, 16, 20, 25, 30]:
        target_len = base_len + target_extra
        if target_len <= len(best_path_ref[0]):
            continue
        if time.time() > deadline:
            break
        
        # Reset best_path_ref to current start_path
        best_path_ref[0] = start_path[:]
        
        # Run DFS with a tighter deadline to avoid spending too much time
        sub_deadline = min(deadline, time.time() + 5.0)
        _dfs_solve(start_path[:], used_mask, sub_deadline, best_path_ref)
        
        if time.time() > deadline:
            break

def _randomized_dfs(start_path, deadline, best_path_ref):
    """
    Randomized DFS: try random orderings of candidates to find diverse paths.
    """
    neighbors = _get_neighbors()
    used_mask = np.zeros(SIZE, dtype=bool)
    for v in start_path:
        used_mask[v] = True
    
    if len(start_path) > len(best_path_ref[0]):
        best_path_ref[0] = start_path[:]
    
    cur = start_path[-1]
    path_len = len(start_path)
    
    # Collect valid extensions
    candidates = []
    for b in range(N):
        cand = int(neighbors[cur, b])
        if used_mask[cand]:
            continue
        valid = True
        for i in range(path_len - 1):
            if (cand ^ start_path[i]).bit_count() == 1:
                valid = False
                break
        if valid:
            candidates.append(cand)
    
    if not candidates:
        return
    
    # Randomize order
    random.shuffle(candidates)
    
    for cand in candidates:
        if time.time() > deadline:
            return
        
        start_path.append(cand)
        used_mask[cand] = True
        
        # Recursively try to extend further
        _randomized_dfs(start_path, deadline, best_path_ref)
        
        start_path.pop()
        used_mask[cand] = False

def construct():
    start_time = time.time()
    
    # Generate a greedy baseline
    greedy_path = _greedy_seed()
    best_path = [greedy_path[:]]
    
    # Try iterative deepening DFS from the greedy path
    _iterative_deepening_dfs(greedy_path, DEADLINE - 10, best_path)
    
    # Try randomized DFS from truncated prefixes of the greedy path
    for trunc_len in [len(greedy_path), len(greedy_path) - 5, len(greedy_path) - 10, len(greedy_path) - 15]:
        if trunc_len <= 0:
            continue
        if time.time() > DEADLINE - 5:
            break
        
        truncated = greedy_path[:trunc_len]
        _randomized_dfs(truncated[:], DEADLINE - 5, best_path)
    
    # Try a few more random starting points
    for _ in range(3):
        if time.time() > DEADLINE - 2:
            break
        
        # Random starting vertex
        start = random.randint(0, SIZE - 1)
        random_path = [start]
        random_used = {start}
        
        # Extend greedily from random start
        while True:
            cur = random_path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in random_used:
                    continue
                valid = True
                for p in random_path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    nxt = v
                    break
            if nxt is None:
                break
            random_path.append(nxt)
            random_used.add(nxt)
        
        if len(random_path) > len(best_path[0]):
            best_path[0] = random_path[:]
        
        # Try to extend with DFS
        _dfs_solve(random_path[:], np.array([v in random_used for v in range(SIZE)], dtype=bool), DEADLINE - 2, best_path)
    
    return best_path[0]
