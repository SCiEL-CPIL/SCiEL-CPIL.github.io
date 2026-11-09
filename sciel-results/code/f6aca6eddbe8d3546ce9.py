import time
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
    
    def try_greedy(start_v, strategy="max_degree"):
        """
        Try to build a snake starting from start_v.
        strategy: 
          - "max_degree": pick neighbor with max available degree in next step
          - "min_degree": pick neighbor with min available degree (Weinstein)
          - "random": pick random valid neighbor
          - "lookahead3": pick neighbor that maximizes the min-degree of its valid neighbors
        """
        path = [start_v]
        used = np.zeros(NUM_VERTICES, dtype=bool)
        used[start_v] = True
        
        # forbidden[v] is True if v is adjacent to any internal vertex in path[:-1]
        forbidden = np.zeros(NUM_VERTICES, dtype=bool)
        
        cur = start_v
        
        while True:
            # Find valid next vertices
            nbrs = neighbors[cur]
            
            valid_next = []
            for n in nbrs:
                if not used[n] and not forbidden[n]:
                    valid_next.append(n)
            
            if not valid_next:
                break
            
            if strategy == "random":
                next_v = valid_next[rng.integers(len(valid_next))]
            elif strategy == "min_degree":
                # Weinstein: pick the vertex with the fewest available moves
                min_score = 100
                best_candidates = []
                for n in valid_next:
                    score = 0
                    for nn in neighbors[n]:
                        if not used[nn] and not forbidden[nn]:
                            score += 1
                    if score < min_score:
                        min_score = score
                        best_candidates = [n]
                    elif score == min_score:
                        best_candidates.append(n)
                next_v = best_candidates[rng.integers(len(best_candidates))]
            elif strategy == "lookahead3":
                # For each candidate n, find valid neighbors nn.
                # For each nn, find valid neighbors nnn.
                # Score n by the minimum number of valid moves from nn.
                best_score = -1
                best_candidates = []
                
                for n in valid_next:
                    # Get valid neighbors of n
                    nn_candidates = []
                    for nn in neighbors[n]:
                        if not used[nn] and not forbidden[nn]:
                            nn_candidates.append(nn)
                    
                    if not nn_candidates:
                        score = 0
                    else:
                        # Score n by the minimum number of valid moves from its valid neighbors
                        min_next_degree = 100
                        for nn in nn_candidates:
                            deg = 0
                            for nnn in neighbors[nn]:
                                if not used[nnn] and not forbidden[nnn]:
                                    deg += 1
                            if deg < min_next_degree:
                                min_next_degree = deg
                        
                        score = min_next_degree
                    
                    if score > best_score:
                        best_score = score
                        best_candidates = [n]
                    elif score == best_score:
                        best_candidates.append(n)
                
                next_v = best_candidates[rng.integers(len(best_candidates))]
            else: # max_degree
                best_score = -1
                best_candidates = []
                for n in valid_next:
                    score = 0
                    for nn in neighbors[n]:
                        if not used[nn] and not forbidden[nn]:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_candidates = [n]
                    elif score == best_score:
                        best_candidates.append(n)
                next_v = best_candidates[rng.integers(len(best_candidates))]
            
            # Add next_v to path
            path.append(next_v)
            used[next_v] = True
            
            # Update forbidden: cur becomes internal, so its neighbors are now forbidden
            for nc in nbrs:
                forbidden[nc] = True
            
            cur = next_v
        
        return path

    def try_greedy_fast(start_v, strategy="max_degree"):
        """
        Fast greedy snake construction using numpy vectorized operations.
        """
        path = [start_v]
        used = np.zeros(NUM_VERTICES, dtype=bool)
        used[start_v] = True
        
        forbidden = np.zeros(NUM_VERTICES, dtype=bool)
        
        cur = start_v
        
        while True:
            nbrs = neighbors[cur]
            valid_mask = (~used[nbrs]) & (~forbidden[nbrs])
            valid_next = nbrs[valid_mask]
            
            if len(valid_next) == 0:
                break
            
            if strategy == "random":
                next_v = int(valid_next[rng.integers(len(valid_next))])
            else:
                # Compute degree scores for all valid candidates
                scores = np.zeros(len(valid_next), dtype=np.int32)
                for i, n in enumerate(valid_next):
                    nbrs_n = neighbors[n]
                    valid_nn_mask = (~used[nbrs_n]) & (~forbidden[nbrs_n])
                    scores[i] = np.sum(valid_nn_mask)
                
                if strategy == "min_degree":
                    min_score = np.min(scores)
                    candidates = valid_next[scores == min_score]
                elif strategy == "lookahead3":
                    la_scores = np.zeros(len(valid_next), dtype=np.int32)
                    for i, n in enumerate(valid_next):
                        nbrs_n = neighbors[n]
                        valid_nn_mask = (~used[nbrs_n]) & (~forbidden[nbrs_n])
                        valid_nn = nbrs_n[valid_nn_mask]
                        if len(valid_nn) == 0:
                            la_scores[i] = 0
                        else:
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
                
                next_v = int(candidates[rng.integers(len(candidates))])
            
            path.append(next_v)
            used[next_v] = True
            
            for nc in nbrs:
                forbidden[nc] = True
            
            cur = next_v
        
        return path

    # Strategy: run many restarts with different starting vertices and strategies
    # Mix fast and slow implementations, prioritizing max_degree and lookahead3
    while time.time() - start_time < MAX_TIME:
        start_v = int(rng.integers(NUM_VERTICES))
        
        r = rng.random()
        if r < 0.4:
            strategy = "max_degree"
            use_fast = True
        elif r < 0.7:
            strategy = "lookahead3"
            use_fast = False  # lookahead3 is slower in fast version due to inner loops
        elif r < 0.85:
            strategy = "min_degree"
            use_fast = True
        else:
            strategy = "random"
            use_fast = True
        
        if use_fast:
            path = try_greedy_fast(start_v, strategy)
        else:
            path = try_greedy(start_v, strategy)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    if best_path is None:
        # Fallback: simple greedy from 0
        path = [0]
        used = {0}
        forbidden = set()
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or v in forbidden:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
            for b in range(N):
                forbidden.add(cur ^ (1 << b))
        best_path = path
    
    return best_path
