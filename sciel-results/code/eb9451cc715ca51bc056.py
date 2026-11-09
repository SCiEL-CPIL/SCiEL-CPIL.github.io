import time
import sys
import random

sys.setrecursionlimit(10000)

N = 256
conflict = [0] * (N + 1)

def compute_conflicts():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a] |= (1 << (b - 1))
                conflict[b] |= (1 << (a - 1))

compute_conflicts()

# Precompute degrees for initial ordering
initial_degrees = [0] * (N + 1)
for i in range(1, N + 1):
    initial_degrees[i] = bin(conflict[i]).count('1')

def popcount(x):
    return bin(x).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit
    
    best_size = 0
    best_mask = 0
    
    initial_mask = (1 << N) - 1
    
    # Greedy heuristic to find a good initial lower bound
    # Sort by degree descending
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: -initial_degrees[x])
    
    greedy_mask = 0
    for v in vertices:
        if (conflict[v] & greedy_mask) == 0:
            greedy_mask |= (1 << (v - 1))
            
    best_size = popcount(greedy_mask)
    best_mask = greedy_mask
    
    # Helper to compute the coloring bound (Clique Cover of Conflict Graph)
    # This gives an upper bound on the size of the MIS in the subgraph induced by 'mask'
    def coloring_bound(mask):
        if mask == 0:
            return 0
        
        # Order vertices by descending degree in the conflict graph restricted to mask
        # This helps form larger cliques early
        verts = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            val = idx + 1
            deg = popcount(conflict[val] & mask)
            verts.append((deg, val))
            temp ^= lsb
            
        # Sort by degree descending
        verts.sort(key=lambda x: -x[0])
        
        color_masks = []
        
        for deg, v in verts:
            v_bit = 1 << (v - 1)
            placed = False
            for i, cm in enumerate(color_masks):
                # v can join color class i if v conflicts with ALL vertices in cm
                # i.e., conflict[v] must contain all bits in cm
                if (conflict[v] & cm) == cm:
                    color_masks[i] = cm | v_bit
                    placed = True
                    break
            if not placed:
                color_masks.append(v_bit)
                
        return len(color_masks)

    def search(candidates_mask, current_mask, current_size):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        # Pruning: trivial upper bound
        if current_size + popcount(candidates_mask) <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Compute coloring bound (upper bound on MIS size in candidates)
        bound = coloring_bound(candidates_mask)
        
        if current_size + bound <= best_size:
            return
            
        # Branching
        # Pick a vertex to branch on. 
        # Heuristic: Pick the vertex with highest degree in the conflict graph?
        # Or just the first one in the ordered list from coloring_bound?
        # Let's pick the vertex with max degree in candidates to maximize pruning effect?
        # Actually, standard Tomita picks vertices in the order they were colored/processed.
        # Let's just pick the first vertex from the 'verts' list generated in coloring_bound 
        # to avoid recomputing, or just pick max degree.
        
        # To avoid recomputing the order, we could pass it, but let's just pick max degree here for simplicity
        # or reuse the logic. Given N=256, recomputing is okay.
        
        max_deg = -1
        v_idx = -1
        temp_mask = candidates_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            val = idx + 1
            deg = popcount(conflict[val] & candidates_mask)
            if deg > max_deg:
                max_deg = deg
                v_idx = idx
            temp_mask ^= lsb
            
        if v_idx == -1:
            return

        v_bit = 1 << v_idx
        v_val = v_idx + 1
        
        # Branch 1: Include v
        # New candidates must not conflict with v
        new_candidates = candidates_mask & ~(conflict[v_val] | v_bit)
        search(new_candidates, current_mask | v_bit, current_size + 1)
        
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude v
        new_candidates = candidates_mask & ~v_bit
        search(new_candidates, current_mask, current_size)

    search(initial_mask, 0, 0)
    
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    if not result:
        return [1]
        
    return result
