import time
import random
import numpy as np

N = 9
MAX_TIME = 118.0
NUM_VERTICES = 1 << N

def construct():
    start_time = time.time()
    
    # Precompute neighbors as a numpy array for fast access
    neighbors = np.empty((NUM_VERTICES, N), dtype=np.uint32)
    for v in range(NUM_VERTICES):
        for b in range(N):
            neighbors[v, b] = v ^ (1 << b)
    
    best_path = None
    best_len = 0
    
    rng = np.random.default_rng(seed=42)
    
    def try_greedy_fast(start_v, strategy="max_degree"):
        """
        Fast greedy snake construction using numpy vectorized operations.
        """
        path = [start_v]
        used = np.zeros(NUM_VERTICES, dtype=bool)
        used[start_v] = True
        
        # forbidden[v] is True if v is adjacent to any internal vertex in path[:-1]
        forbidden = np.zeros(NUM_VERTICES, dtype=bool)
        
        cur = start_v
        
        while True:
            # Find valid next vertices using numpy
            nbrs = neighbors[cur]
            valid_mask = (~used[nbrs]) & (~forbidden[nbrs])
            valid_next = nbrs[valid_mask]
            
            if len(valid_next) == 0:
                break
            
            if strategy == "random":
                next_v = valid_next[rng.integers(len(valid_next))]
            else:
                # Compute degree scores for all valid candidates at once
                # For each candidate n, count its valid neighbors
                scores = np.zeros(len(valid_next), dtype=np.int32)
                for i, n in enumerate(valid_next):
                    nbrs_n = neighbors[n]
                    # A neighbor nn of n is valid if:
                    # - not used
                    # - not forbidden
                    # - not adjacent to cur (but as analyzed, only cur itself is adjacent, and it's used)
                    # So just check used and forbidden
                    valid_nn_mask = (~used[nbrs_n]) & (~forbidden[nbrs_n])
                    scores[i] = np.sum(valid_nn_mask)
                
                if strategy == "min_degree":
                    min_score = np.min(scores)
                    candidates = valid_next[scores == min_score]
                elif strategy == "lookahead3":
                    # For lookahead, score by min degree of next-level neighbors
                    la_scores = np.zeros(len(valid_next), dtype=np.int32)
                    for i, n in enumerate(valid_next):
                        nbrs_n = neighbors[n]
                        valid_nn_mask = (~used[nbrs_n]) & (~forbidden[nbrs_n])
                        valid_nn = nbrs_n[valid_nn_mask]
                        if len(valid_nn) == 0:
                            la_scores[i] = 0
                        else:
                            # For each valid nn, count its valid neighbors
                            nn_degrees = np.zeros(len(valid_nn), dtype=np.int32)
                            for j, nn in enumerate(valid_nn):
                                nbrs_nn = neighbors[nn]
                                valid_nnn_mask = (~used[nbrs_nn]) & (~forbidden[nbrs_nn])
                                nn_degrees[j] = np.sum(valid_nnn_mask)
                            la_scores[i] = np.min(nn_degrees)
                    max_score = np.max(la_scores)
                    candidates = valid_next[la_scores == max_score]
                else:  # max_degree
                    max_score = np.max(scores)
                    candidates = valid_next[scores == max_score]
                
                next_v = candidates[rng.integers(len(candidates))]
            
            # Add next_v to path
            path.append(next_v)
            used[next_v] = True
            
            # Update forbidden: cur becomes internal, so its neighbors are now forbidden
            forbidden[nbrs] = True
            
            cur = next_v
        
        return path
    
    def refine_path(path):
        """
        Try to extend or improve the path by backtracking at critical points.
        Returns a potentially longer path.
        """
        if not path or len(path) < 2:
            return path
        
        # Try to extend from the end by trying different last moves
        # This is a simple local search: try to find an alternative at some point
        # that leads to a longer path
        
        # For now, just return the path as-is if it's already good
        # A more sophisticated refinement would try to swap choices
        return path
    
    # Strategy: run many restarts with different starting vertices and strategies
    # Focus on fast strategies since we need many restarts in the time budget
    
    while time.time() - start_time < MAX_TIME - 1.0:
        # Random start
        start_v = rng.integers(NUM_VERTICES)
        
        # Random strategy choice - weight towards faster strategies
        r = rng.random()
        if r < 0.6:
            strategy = "max_degree"
        elif r < 0.85:
            strategy = "min_degree"
        else:
            strategy = "lookahead3"
        
        path = try_greedy_fast(start_v, strategy)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    if best_path is None:
        # Fallback: simple greedy from 0
        path = [0]
        used = np.zeros(NUM_VERTICES, dtype=bool)
        used[0] = True
        forbidden = np.zeros(NUM_VERTICES, dtype=bool)
        cur = 0
        while True:
            nbrs = neighbors[cur]
            valid_mask = (~used[nbrs]) & (~forbidden[nbrs])
            valid_next = nbrs[valid_mask]
            if len(valid_next) == 0:
                break
            next_v = valid_next[0]
            path.append(next_v)
            used[next_v] = True
            forbidden[nbrs] = True
            cur = next_v
        best_path = path
    
    return best_path
