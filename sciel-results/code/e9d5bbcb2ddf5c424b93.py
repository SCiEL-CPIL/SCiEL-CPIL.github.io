import numpy as np
from itertools import permutations
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Classic construction
    classic_word = _classic_construct(n)
    best_word = classic_word
    best_len = len(classic_word)
    
    # For n=7, we attempt TSP optimization to try to beat 5913
    if n >= 4 and n <= 7:
        if n == 7:
            t = _tsp_construct(n, deadline)
            if t is not None:
                if len(t) < best_len:
                    best_word = t
                    best_len = len(t)
        elif n < 7:
            # For n<7, classic is optimal, but TSP might find something
            t = _tsp_construct(n, deadline)
            if t is not None:
                if len(t) < best_len:
                    best_word = t
                    best_len = len(t)

    return best_word

def _classic_construct(n):
    """Classic recursive superpermutation construction.
    For n=7, produces length 5913.
    """
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new and start of block
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _tsp_construct(n, deadline):
    """TSP-based construction: order permutations to maximize total overlap.
    
    The key insight: if we order all n! permutations in a sequence, the total
    length of the superpermutation is n + sum over consecutive pairs of (n - overlap).
    So minimizing total length = maximizing total overlap = TSP on overlap matrix.
    
    We use nearest-neighbor initialization + 2-opt local search.
    """
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # Convert to strings for fast overlap computation
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute overlap matrix: overlap[i][j] = longest suffix of perms[i] 
    # that is a prefix of perms[j]
    # For n=7, max overlap is 6
    overlap = np.zeros((N, N), dtype=np.int8)
    
    # Efficient overlap computation using string operations
    for i in range(N):
        si = perm_strs[i]
        for j in range(N):
            sj = perm_strs[j]
            # Check overlaps from longest to shortest
            for l in range(n - 1, 0, -1):
                if si[-l:] == sj[:l]:
                    overlap[i, j] = l
                    break
    
    if time.time() > deadline - 30:
        return None
    
    # Build cost matrix: cost[i][j] = n - overlap[i][j]
    # We want to find a Hamiltonian path minimizing total cost
    # Equivalent to maximizing total overlap
    
    # Nearest neighbor initialization with multiple random starts
    best_path = None
    best_cost = float('inf')
    
    num_starts = min(50, N)  # Multiple random starting points
    rng = np.random.default_rng(42)
    
    for start in range(num_starts):
        if time.time() > deadline - 20:
            break
            
        # Start from a random permutation
        curr = start if start < N else rng.integers(0, N)
        visited = np.zeros(N, dtype=bool)
        visited[curr] = True
        path = [curr]
        
        for _ in range(N - 1):
            if time.time() > deadline - 20:
                break
            # Find unvisited node with maximum overlap (minimum cost)
            unvisited_mask = ~visited
            unvisited_mask[curr] = False
            unvisited_indices = np.where(unvisited_mask)[0]
            
            if len(unvisited_indices) == 0:
                break
            
            # Get overlaps from curr to all unvisited
            ovs = overlap[curr, unvisited_indices]
            best_idx = np.argmax(ovs)
            next_idx = unvisited_indices[best_idx]
            
            visited[next_idx] = True
            path.append(next_idx)
            curr = next_idx
        
        if len(path) != N:
            continue
        
        # Calculate cost of this path
        cost = 0
        for k in range(N - 1):
            cost += n - overlap[path[k], path[k + 1]]
        
        if cost < best_cost:
            best_cost = cost
            best_path = path[:]
    
    if best_path is None:
        return None
    
    # 2-opt local search improvement
    # For a path (not cycle), 2-opt reverses a segment
    # Edge changes: (i-1 -> i) and (j -> j+1) become (i-1 -> j) and (i -> j+1)
    # where we reverse path[i:j+1]
    
    path = best_path[:]
    improved = True
    max_iters = 200
    iter_count = 0
    
    while improved and iter_count < max_iters and time.time() < deadline - 5:
        improved = False
        iter_count += 1
        
        for i in range(1, N - 1):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, N):
                # Old edges: (i-1 -> i) and (j -> j+1) if j < N-1
                # New edges: (i-1 -> j) and (i -> j+1) if j < N-1
                
                old_cost = 0
                new_cost = 0
                
                # Edge (i-1 -> i)
                old_cost += n - overlap[path[i-1], path[i]]
                new_cost += n - overlap[path[i-1], path[j]]
                
                # Edge (j -> j+1) if exists
                if j < N - 1:
                    old_cost += n - overlap[path[j], path[j+1]]
                    new_cost += n - overlap[path[i], path[j+1]]
                
                if new_cost < old_cost:
                    path[i:j+1] = path[i:j+1][::-1]
                    improved = True
    
    # Build the superpermutation string
    s = perm_strs[path[0]]
    for k in range(1, N):
        p_prev = perm_strs[path[k-1]]
        p_curr = perm_strs[path[k]]
        # Find overlap
        ov = 0
        for l in range(n - 1, 0, -1):
            if p_prev[-l:] == p_curr[:l]:
                ov = l
                break
        s += p_curr[ov:]
    
    # Verify validity: check all n! permutations appear as windows
    windows = set()
    for i in range(len(s) - n + 1):
        w = tuple(map(int, s[i:i+n]))
        windows.add(w)
    
    if len(windows) == N:
        return s
    else:
        return None
