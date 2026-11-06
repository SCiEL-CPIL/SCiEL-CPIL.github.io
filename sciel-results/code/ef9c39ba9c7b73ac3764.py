import time
import random
import numpy as np
from collections import deque

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 115.0  # Safety margin: killed at 150s
    
    # Precompute neighbors using bit operations
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute popcount for fast Hamming distance
    popcount = np.zeros(MAX_V, dtype=np.uint8)
    for i in range(MAX_V):
        popcount[i] = bin(i).count('1')
    
    # Precompute adjacency matrix as a set of frozensets for fast lookup? No, use boolean matrix or bitsets.
    # For N=9, MAX_V=512. A 512x512 boolean matrix is small.
    adj_matrix = np.zeros((MAX_V, MAX_V), dtype=bool)
    for v in range(MAX_V):
        for nb in neighbors[v]:
            adj_matrix[v, nb] = True
    
    # Precompute dist >= 2 mask? 
    # dist2_mask[v] is a boolean array where dist2_mask[v, u] is True if dist(v,u) >= 2
    # Actually, we want to check if a new vertex is compatible with all existing vertices.
    # Let's precompute a matrix 'compatible' where compatible[v, u] is True if dist(v,u) >= 2.
    # Wait, if dist is 1, they are adjacent. If dist is 0, they are same.
    # We need dist >= 2 for non-consecutive.
    # So compatible[v, u] = (popcount[v ^ u] >= 2)
    
    # We can use bitsets for the set of used vertices to quickly check compatibility.
    # But Python integers can act as bitsets of arbitrary size.
    # Let's represent the set of used vertices as a Python integer bitmask.
    # used_mask: bit i is 1 if vertex i is in the path.
    
    # Precompute for each vertex v, a bitmask of vertices u such that dist(v, u) >= 2.
    # This allows us to check: if (used_mask & ~compatible_mask[v]) == 0, then v is compatible with all used vertices.
    # Wait, we need to exclude the immediate neighbors (predecessor/successor) from this check?
    # No, the condition is: for all u in path, if u is not adjacent to v in the path, then dist(v, u) >= 2.
    # If u is adjacent to v in the path, dist(v, u) = 1 is allowed.
    # So, when checking if we can add v to the end of the path (current last is last_v):
    # v must be adjacent to last_v.
    # v must be compatible with all u in path except last_v.
    # So we need: (used_mask_without_last_v & ~compatible_mask[v]) == 0.
    # used_mask_without_last_v = used_mask ^ (1 << last_v)
    
    compatible_mask = [0] * MAX_V
    for v in range(MAX_V):
        mask = 0
        for u in range(MAX_V):
            if popcount[v ^ u] >= 2:
                mask |= (1 << u)
        compatible_mask[v] = mask
    
    best_path = []
    best_len = 0
    
    def greedy_extend(path, used_mask, randomize=True):
        """Greedily extend a path using Warnsdorff-like heuristic with bitmasks."""
        if time.time() > deadline:
            return path, used_mask
        
        while True:
            if time.time() > deadline:
                return path, used_mask
            
            cur = path[-1]
            
            # Find valid candidates: unused vertices adjacent to cur
            # Candidate nb must satisfy:
            # 1. nb not in used_mask
            # 2. (used_mask ^ (1 << cur)) & ~compatible_mask[nb] == 0
            #    i.e., nb is compatible with all used vertices except cur.
            
            candidates = []
            for nb in neighbors[cur]:
                if (used_mask >> nb) & 1:
                    continue
                # Check compatibility with all other used vertices
                # used_mask without cur
                mask_others = used_mask ^ (1 << cur)
                if (mask_others & ~compatible_mask[nb]) == 0:
                    # Compute future degree: how many valid extensions nb would have
                    # This is expensive. Let's just use a simpler heuristic or skip for now.
                    # Warnsdorff: prefer vertices with fewer remaining options.
                    # Let's estimate remaining options by counting neighbors of nb that are compatible with (used_mask | (1 << nb)) except nb's predecessor (cur).
                    # Actually, just count neighbors of nb that are not in (used_mask | (1 << nb)).
                    # A rough estimate is enough.
                    new_used = used_mask | (1 << nb)
                    deg = 0
                    for nn in neighbors[nb]:
                        if (new_used >> nn) & 1:
                            continue
                        # Check if nn is compatible with all vertices in new_used except nb
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
                # Pick from top few
                k = min(3, len(candidates))
                chosen = random.choice(candidates[:k])[1]
            else:
                chosen = candidates[0][1]
            
            # Add chosen
            path.append(chosen)
            used_mask |= (1 << chosen)
        
    def local_search_improve(path, used_mask, max_iters=10):
        """Try to improve a path by local search: try to extend at ends."""
        if time.time() > deadline:
            return path, used_mask
        
        best = path[:]
        best_mask = used_mask
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            n = len(best)
            
            # Try extending the path at either end
            # Try to find a vertex that can be prepended
            first = best[0]
            for nb in neighbors[first]:
                if (best_mask >> nb) & 1:
                    continue
                # Check if nb can be prepended: nb adjacent to first (yes), 
                # and nb at distance >= 2 from all vertices in best[1:]
                mask_others = best_mask ^ (1 << first)
                if (mask_others & ~compatible_mask[nb]) == 0:
                    new_path = [nb] + best
                    new_mask = best_mask | (1 << nb)
                    best = new_path
                    best_mask = new_mask
                    improved = True
                    break
            if improved:
                continue
            
            # Try to find a vertex that can be appended
            last = best[-1]
            for nb in neighbors[last]:
                if (best_mask >> nb) & 1:
                    continue
                # Check if nb can be appended: nb adjacent to last (yes), 
                # and nb at distance >= 2 from all vertices in best[:-1]
                mask_others = best_mask ^ (1 << last)
                if (mask_others & ~compatible_mask[nb]) == 0:
                    new_path = best + [nb]
                    new_mask = best_mask | (1 << nb)
                    best = new_path
                    best_mask = new_mask
                    improved = True
                    break
            if improved:
                continue
            
            # Try to find a vertex to insert? 
            # As analyzed, inserting between adjacent vertices is impossible in a hypercube.
            # So we can only extend at ends.
            
            break
        
        return best, best_mask
    
    # Main search: multiple random restarts with greedy + local improvement
    random.seed(12345)
    
    # Try many starting vertices
    start_vertices = list(range(MAX_V))
    random.shuffle(start_vertices)
    
    num_restarts = 500  # We'll limit by time
    
    for start_idx in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        start = start_vertices[start_idx]
        
        path = [start]
        used_mask = 1 << start
        
        # Greedy extension with randomization
        path, used_mask = greedy_extend(path, used_mask, randomize=True)
        
        # Local search improvement
        path, used_mask = local_search_improve(path, used_mask, max_iters=10)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Final check
    if not best_path:
        return [0]
    
    return best_path
