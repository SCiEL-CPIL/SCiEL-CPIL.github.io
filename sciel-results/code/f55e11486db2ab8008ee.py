import time
import random
import itertools
import numpy as np

def compute_overlap_matrix(perms):
    """
    Compute overlap matrix for a list of permutations.
    overlap[i][j] is the maximum k such that suffix of perms[i] of length k 
    equals prefix of perms[j] of length k.
    """
    n = len(perms)
    overlap = [[0] * n for _ in range(n)]
    for i in range(n):
        pi = perms[i]
        for j in range(n):
            pj = perms[j]
            # Max possible overlap is len(pi)
            max_k = len(pi)
            best = 0
            for k in range(1, max_k + 1):
                if pi[-k:] == pj[:k]:
                    best = k
            overlap[i][j] = best
    return overlap

def greedy_path(perms, overlap_matrix, n_restarts=10):
    """
    Find a good ordering of perms to maximize total overlap.
    Returns the list of indices of the optimal path found.
    Uses stochastic selection to explore different paths.
    """
    n = len(perms)
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    ov_mat = np.array(overlap_matrix)
    best_path = None
    best_score = -1
    
    for r in range(n_restarts):
        # Random start
        start = random.randint(0, n - 1)
        path = [start]
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        current = start
        current_score = 0
        
        for _ in range(n - 1):
            # Get overlaps from current to all
            scores = ov_mat[current].copy()
            scores[visited] = -1 # Mask visited
            # Add small noise to break ties and explore
            scores += np.random.uniform(0, 0.1, size=n)
            
            next_node = np.argmax(scores)
            
            visited[next_node] = True
            path.append(next_node)
            current_score += ov_mat[current][next_node]
            current = next_node
        
        if current_score > best_score:
            best_score = current_score
            best_path = path
            
    return best_path

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract distinct (m-1) permutations from word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Compute overlap matrix
        if time.time() > deadline:
            break
            
        overlap_matrix = compute_overlap_matrix(perms)
        
        # Determine restarts based on size
        # Numpy makes it fast, so we can afford more restarts
        # For n=7, N=720. 1000 restarts * 720 steps * 720 ops ~ 500M ops?
        # Numpy argmax is fast. Let's try 100 restarts first, maybe increase if time permits.
        # Actually, 100 restarts should be safe.
        restarts = 100 if len(perms) > 500 else 200
        
        path = greedy_path(perms, overlap_matrix, n_restarts=restarts)
        ordered_perms = [perms[i] for i in path]
        
        blocks = [list(p) + [m] + list(p) for p in ordered_perms]
        new = []
        if blocks:
            new += blocks[0]
            for k in range(1, len(blocks)):
                # Use precomputed overlap for merging
                ov = overlap_matrix[path[k-1]][path[k]]
                new += blocks[k][ov:]
        
        word = new
        
        if time.time() > deadline:
            break

    return "".join(map(str, word))
