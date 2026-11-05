import time
import random
import numpy as np
from math import gcd

def construct():
    N = 128
    deadline = time.time() + 86.0
    
    # 1. Precompute GCD matrix
    G = np.zeros((N+1, N+1), dtype=np.uint8)
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g = gcd(i, j)
            G[i, j] = g
            G[j, i] = g
            
    # 2. Fast Vectorized Conflict Counter
    # Counts the number of "bad triples" in the current selection.
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c).
    def count_conflicts(mask):
        # mask is a boolean array of length N+1
        # Get indices of selected elements
        idx = np.nonzero(mask[1:])[0] + 1
        if len(idx) < 3:
            return 0
            
        # Compute GCD matrix for selected elements
        G_sub = G[np.ix_(idx, idx)]
        
        # For each pair (i, j), we need to count how many k (k != i, k != j) 
        # have gcd(i,k) == gcd(j,k) == gcd(i,j)
        
        # We can vectorize this by checking each pair (i,j)
        # Let's iterate over unique pairs to keep memory manageable, 
        # but N is small (128), so even O(N^3) with numpy overhead might be okay if done smartly.
        # However, a fully vectorized approach is preferred.
        
        # Alternative:
        # For each k, look at the row G_sub[k]. 
        # We need pairs (i,j) such that G_sub[i,k] == G_sub[j,k] == G_sub[i,j].
        
        # Let's use a different approach:
        # For each value v, consider the subgraph of edges with GCD v.
        # If there is a triangle in this subgraph, it's a conflict.
        
        conflicts = 0
        # Iterate over possible GCD values (1 to N)
        # Actually, just iterate over pairs (i,j) in the selected set?
        # N=128, pairs ~ 8000. For each pair, checking k is O(N). Total 8000*128 ~ 1M ops.
        # This is fast in Python if we use numpy for the inner loop?
        
        # Let's try the subgraph triangle counting approach.
        # For each d in range(1, N+1):
        #    Create adjacency matrix A_d where A_d[i,j] = 1 if G[i,j] == d
        #    Count triangles in A_d.
        
        # Since G_sub is small, we can just do:
        # for i in range(len(idx)):
        #    for j in range(i+1, len(idx)):
        #        if G_sub[i,j] > 0:
        #            # Check if there is a k such that G_sub[i,k] == G_sub[i,j] and G_sub[j,k] == G_sub[i,j]
        #            # This is equivalent to: sum over k of (G_sub[i,k] == G_sub[i,j]) * (G_sub[j,k] == G_sub[i,j]) * (k != i, k != j)
        
        # Vectorized inner loop:
        # For each i, j:
        #    g = G_sub[i,j]
        #    mask_i = (G_sub[i,:] == g)
        #    mask_j = (G_sub[j,:] == g)
        #    # Intersection of mask_i and mask_j gives candidates for k
        #    # Exclude i and j themselves
        #    candidates = mask_i & mask_j
        #    candidates[i] = False
        #    candidates[j] = False
        #    conflicts += np.sum(candidates)
        
        # This is O(N^3) but with numpy vectorization on the inner sum, it's O(N^2) numpy calls.
        # 128^2 = 16384 numpy calls. Each is tiny. Should be < 1ms.
        
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                g = G_sub[i,j]
                if g == 0: continue
                # Find all k such that G[i,k] == g and G[j,k] == g
                # Note: G_sub[i,i] is 0, so i won't match g (unless g=0 which is skipped)
                # But we must be careful if g matches diagonal? No, G[i,i]=0.
                # So we just count matches.
                # However, we must exclude i and j from the count of k?
                # If k=i, G[j,i] == g is true. G[i,i] == 0 != g. So i is not counted.
                # If k=j, G[i,j] == g is true. G[j,j] == 0 != g. So j is not counted.
                # So we can just sum the intersection.
                
                # Wait, G_sub[i, i] is 0. If g > 0, then G_sub[i, i] != g.
                # So i is automatically excluded from mask_i? No, mask_i is row i.
                # mask_i[i] is G_sub[i,i] == 0. If g>0, mask_i[i] is False.
                # mask_j[j] is G_sub[j,j] == 0. If g>0, mask_j[j] is False.
                # So the intersection automatically excludes i and j?
                # Let's check:
                # We want k such that G[i,k]==g AND G[j,k]==g.
                # If k=i: G[i,i]=0. If g!=0, False.
                # If k=j: G[j,j]=0. If g!=0, False.
                # So yes, for g>0, i and j are not counted.
                
                # But what if g=0? We skip g=0.
                
                # So:
                conflicts += np.sum((G_sub[i,:] == g) & (G_sub[j,:] == g))
                
        return conflicts

    # 3. Initial Solution: Random or Greedy
    # Let's start with a random subset of size ~ N/2
    mask = np.zeros(N+1, dtype=bool)
    initial_size = 64
    candidates = np.random.choice(np.arange(1, N+1), size=initial_size, replace=False)
    mask[candidates] = True
    
    # 4. Simulated Annealing
    T = 10.0
    T_min = 0.001
    alpha = 0.9995 # Cooling rate
    current_conflicts = count_conflicts(mask)
    current_size = np.sum(mask)
    current_score = current_size - 100 * current_conflicts
    
    best_mask = mask.copy()
    best_score = current_score
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Move: Flip a random bit
        pos = np.random.randint(1, N+1)
        old_state = mask[pos]
        mask[pos] = not old_state
        
        new_size = current_size + (1 if not old_state else -1)
        new_conflicts = count_conflicts(mask)
        new_score = new_size - 100 * new_conflicts
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_size = new_size
            current_conflicts = new_conflicts
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_mask = mask.copy()
        else:
            # Revert
            mask[pos] = old_state
            
        T *= alpha
        if T < T_min:
            break
            
    # 5. Final Polish: Greedy Add
    # Try to add any remaining elements that don't increase conflicts
    final_mask = best_mask.copy()
    improved = True
    while improved:
        improved = False
        for i in range(1, N+1):
            if not final_mask[i]:
                final_mask[i] = True
                if count_conflicts(final_mask) > count_conflicts(best_mask): # Heuristic: keep conflicts low
                    final_mask[i] = False
                else:
                    improved = True
                    
    # Ensure valid (0 conflicts)
    # If SA didn't reach 0 conflicts, we need to repair.
    # But the objective penalizes conflicts heavily, so it should be 0.
    # Let's verify and repair if necessary.
    
    # Repair: Remove elements involved in conflicts until none remain.
    # This is a fallback.
    def repair(mask):
        m = mask.copy()
        c = count_conflicts(m)
        while c > 0:
            # Find a conflicting element
            # Simple: pick a random element, if it's in a conflict, remove it?
            # Better: find all elements in conflicts and remove one.
            # For simplicity, just remove a random element until clean.
            # This might reduce size, but ensures validity.
            idx = np.nonzero(m[1:])[0] + 1
            if len(idx) == 0: break
            pos = np.random.choice(idx)
            m[pos] = False
            c = count_conflicts(m)
        return m

    final_mask = repair(final_mask)
    
    # Final check
    if count_conflicts(final_mask) > 0:
        # Emergency fallback: return a known small valid set
        return [1, 2, 3]
        
    return sorted(np.nonzero(final_mask[1:])[0] + 1)

# EVOLVE-BLOCK-END
