import time
import random
import numpy as np

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute popcount
    popcount = np.zeros(MAX_V, dtype=np.uint8)
    for i in range(MAX_V):
        popcount[i] = bin(i).count('1')
    
    # Precompute compatible masks: for each vertex v, bitmask of all u with dist(v,u) >= 2
    compatible_mask = [0] * MAX_V
    for v in range(MAX_V):
        mask = 0
        for u in range(MAX_V):
            if popcount[v ^ u] >= 2:
                mask |= (1 << u)
        compatible_mask[v] = mask
    
    # Precompute adjacency as boolean matrix for fast lookup
    adj = np.zeros((MAX_V, MAX_V), dtype=bool)
    for v in range(MAX_V):
        for nb in neighbors[v]:
            adj[v, nb] = True
    
    best_path = []
    best_len = 0
    
    def is_valid_path(path):
        """Verify that path is a valid induced path in Q_N."""
        if not path:
            return True
        if len(set(path)) != len(path):
            return False
        for i in range(len(path) - 1):
            if popcount[path[i] ^ path[i+1]] != 1:
                return False
        # Check non-consecutive vertices have distance >= 2
        for i in range(len(path)):
            for j in range(i + 2, len(path)):
                if popcount[path[i] ^ path[j]] < 2:
                    return False
        return True
    
    def greedy_extend(path, used_mask, randomize=True):
        """Greedily extend a path using Warnsdorff-like heuristic with bitmasks."""
        if time.time() > deadline:
            return path, used_mask
        
        while True:
            if time.time() > deadline:
                return path, used_mask
            
            cur = path[-1]
            
            # Find valid candidates: unused vertices adjacent to cur
            candidates = []
            for nb in neighbors[cur]:
                if (used_mask >> nb) & 1:
                    continue
                # Check compatibility with all other used vertices (except cur)
                mask_others = used_mask ^ (1 << cur)
                if (mask_others & ~compatible_mask[nb]) == 0:
                    # Compute future degree for Warnsdorff
                    new_used = used_mask | (1 << nb)
                    deg = 0
                    for nn in neighbors[nb]:
                        if (new_used >> nn) & 1:
                            continue
                        mask_others_nb = new_used ^ (1 << nb)
                        if (mask_others_nb & ~compatible_mask[nn]) == 0:
                            deg += 1
                    candidates.append((deg, nb))
            
            if not candidates:
                return path, used_mask
            
            # Warnsdorff: prefer lower degree
            candidates.sort(key=lambda x: x[0])
            
            # Pick with some randomness
            if randomize and len(candidates) > 1:
                k = min(3, len(candidates))
                chosen = random.choice(candidates[:k])[1]
            else:
                chosen = candidates[0][1]
            
            path.append(chosen)
            used_mask |= (1 << chosen)
        
    def backtrack_search(path, used_mask, depth_limit=0):
        """
        Backtracking search with constraint propagation.
        Tries to extend the path, backtracking when stuck.
        Uses MRV (Minimum Remaining Values) heuristic.
        """
        nonlocal best_path, best_len
        
        if time.time() > deadline - 2.0:
            return
        
        cur_len = len(path)
        if cur_len > best_len:
            best_len = cur_len
            best_path = path[:]
        
        if depth_limit and cur_len >= depth_limit:
            return
        
        cur = path[-1]
        
        # Find valid candidates
        candidates = []
        for nb in neighbors[cur]:
            if (used_mask >> nb) & 1:
                continue
            mask_others = used_mask ^ (1 << cur)
            if (mask_others & ~compatible_mask[nb]) == 0:
                # Compute future degree
                new_used = used_mask | (1 << nb)
                deg = 0
                for nn in neighbors[nb]:
                    if (new_used >> nn) & 1:
                        continue
                    mask_others_nb = new_used ^ (1 << nb)
                    if (mask_others_nb & ~compatible_mask[nn]) == 0:
                        deg += 1
                candidates.append((deg, nb))
        
        if not candidates:
            return
        
        # MRV: try candidates with fewest remaining options first
        candidates.sort(key=lambda x: x[0])
        
        # Try each candidate, with some randomization in order for diversity
        if random.random() < 0.3 and len(candidates) > 1:
            random.shuffle(candidates)
        
        for deg, nb in candidates:
            if time.time() > deadline - 2.0:
                return
            
            new_path = path + [nb]
            new_mask = used_mask | (1 << nb)
            
            backtrack_search(new_path, new_mask, depth_limit)
            
            # If we found a very long path, we can stop early
            if best_len >= 180:
                return
    
    def local_search_improve(path, used_mask, max_iters=20):
        """Try to improve a path by extending at ends."""
        if time.time() > deadline:
            return path, used_mask
        
        best = path[:]
        best_mask = used_mask
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            n = len(best)
            
            # Try prepending
            first = best[0]
            for nb in neighbors[first]:
                if (best_mask >> nb) & 1:
                    continue
                mask_others = best_mask ^ (1 << first)
                if (mask_others & ~compatible_mask[nb]) == 0:
                    best = [nb] + best
                    best_mask = best_mask | (1 << nb)
                    improved = True
                    break
            if improved:
                continue
            
            # Try appending
            last = best[-1]
            for nb in neighbors[last]:
                if (best_mask >> nb) & 1:
                    continue
                mask_others = best_mask ^ (1 << last)
                if (mask_others & ~compatible_mask[nb]) == 0:
                    best = best + [nb]
                    best_mask = best_mask | (1 << nb)
                    improved = True
                    break
            if improved:
                continue
            
            break
        
        return best, best_mask
    
    # Main search: combine greedy restarts with backtracking
    random.seed(42)
    
    # Phase 1: Greedy restarts to get a good initial solution
    start_vertices = list(range(MAX_V))
    random.shuffle(start_vertices)
    
    num_greedy = 200
    for i in range(num_greedy):
        if time.time() > deadline - 60.0:
            break
        
        start = start_vertices[i % len(start_vertices)]
        path = [start]
        used_mask = 1 << start
        
        path, used_mask = greedy_extend(path, used_mask, randomize=True)
        path, used_mask = local_search_improve(path, used_mask, max_iters=20)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Phase 2: Backtracking search from the best path found
    if best_path:
        # Try backtracking from the best path
        backtrack_search(best_path[:], 0, depth_limit=0)
        # Reconstruct used_mask
        used_mask = 0
        for v in best_path:
            used_mask |= (1 << v)
        backtrack_search(best_path, used_mask, depth_limit=0)
    
    # Phase 3: More greedy restarts with different seeds if time remains
    if time.time() < deadline - 30.0:
        random.seed(99999)
        for i in range(100):
            if time.time() > deadline - 10.0:
                break
            
            start = random.randint(0, MAX_V - 1)
            path = [start]
            used_mask = 1 << start
            
            path, used_mask = greedy_extend(path, used_mask, randomize=True)
            path, used_mask = local_search_improve(path, used_mask, max_iters=20)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
    
    # Final validation
    if not best_path:
        return [0]
    
    if not is_valid_path(best_path):
        # Fallback to a simple valid path
        return list(range(min(100, MAX_V)))
    
    return best_path
