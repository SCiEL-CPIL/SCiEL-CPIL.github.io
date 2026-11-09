import time
import random
import numpy as np

N = 9
MAX_TIME = 118.0
NUM_VERTICES = 1 << N

def construct():
    start_time = time.time()
    
    # Precompute neighbors
    neighbors = np.empty((NUM_VERTICES, N), dtype=np.uint32)
    for v in range(NUM_VERTICES):
        for b in range(N):
            neighbors[v, b] = v ^ (1 << b)
    
    best_path = None
    best_len = 0
    
    rng = np.random.default_rng(seed=42)
    
    def try_greedy(start_v, strategy="max_degree"):
        path = [start_v]
        used = np.zeros(NUM_VERTICES, dtype=bool)
        used[start_v] = True
        forbidden = np.zeros(NUM_VERTICES, dtype=bool)
        cur = start_v
        
        while True:
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
                best_score = -1
                best_candidates = []
                
                for n in valid_next:
                    nn_candidates = []
                    for nn in neighbors[n]:
                        if not used[nn] and not forbidden[nn]:
                            nn_candidates.append(nn)
                    
                    if not nn_candidates:
                        score = 0
                    else:
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
            else:  # max_degree
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
            
            path.append(next_v)
            used[next_v] = True
            
            for nc in nbrs:
                forbidden[nc] = True
            
            cur = next_v
        
        return path
    
    def refine_path(path):
        """
        Try to extend or rearrange a path to make it longer.
        This is a simple local search: try to find a different starting point
        or a different greedy choice that yields a longer path.
        """
        if not path:
            return path
        
        # Try to extend from the end
        cur = path[-1]
        used = np.zeros(NUM_VERTICES, dtype=bool)
        for v in path:
            used[v] = True
        
        # Reconstruct forbidden from the path
        forbidden = np.zeros(NUM_VERTICES, dtype=bool)
        for i in range(len(path) - 1):
            v = path[i]
            for nc in neighbors[v]:
                forbidden[nc] = True
        
        # Try to extend
        extended = list(path)
        while True:
            nbrs = neighbors[cur]
            valid_next = []
            for n in nbrs:
                if not used[n] and not forbidden[n]:
                    valid_next.append(n)
            
            if not valid_next:
                break
            
            # Use min_degree strategy for extension
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
            extended.append(next_v)
            used[next_v] = True
            
            for nc in nbrs:
                forbidden[nc] = True
            
            cur = next_v
        
        return extended
    
    # Strategy: run many restarts with different starting vertices and strategies
    while time.time() - start_time < MAX_TIME:
        start_v = rng.integers(NUM_VERTICES)
        
        r = rng.random()
        if r < 0.4:
            strategy = "max_degree"
        elif r < 0.7:
            strategy = "lookahead3"
        else:
            strategy = "min_degree"
        
        path = try_greedy(start_v, strategy)
        
        # Try to refine
        if len(path) > best_len:
            refined = refine_path(path)
            if len(refined) > best_len:
                best_len = len(refined)
                best_path = refined
            elif len(path) > best_len:
                best_len = len(path)
                best_path = path
    
    if best_path is None:
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
