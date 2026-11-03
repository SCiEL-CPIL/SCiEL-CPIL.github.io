"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Ensure int64 for speed
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    
    # Precompute row sums of a for fast cost delta calculation
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # If we swap p[i] and p[j], the change is:
    # - old: a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]] + sum_{k!=i,j} (a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]]) + ...
    # This is complex to update incrementally without storing the full cost matrix or similar.
    # However, for 256, a full cost calculation is O(n^2) = 65k ops.
    # 150s / 65k ops is ~2300 full cost evals per second if pure Python.
    # With NumPy vectorization, we can do much better.
    
    # Let's implement a fast cost function using NumPy
    def calc_cost(p):
        # p is a list or array of locations
        # b[p[:, None], p[None, :]] gives the distance matrix for the permutation
        # cost = np.sum(a * b[np.ix_(p, p)])
        # This creates a n x n matrix, which is 256*256 = 65k elements. Very fast.
        return np.sum(a * b[np.ix_(p, p)])
    
    # Initial heuristic: Rank matching
    # Heavy flows (high row sum in a) should go to central locations (low row sum in b)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_idx = np.argsort(-a_sums, kind='stable')
    # Sort locations by ascending distance sum
    central_idx = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx
    p = p.tolist()
    
    current_cost = calc_cost(p)
    best_p = p[:]
    best_cost = current_cost
    
    # Local Search: 2-opt style swaps
    # Random restarts if time permits
    
    rng = random.Random(42)
    
    # We can afford many iterations.
    # Let's try a simple simulated annealing or hill climbing.
    # Given the size (256), 2-opt is O(n^2) candidates per step.
    # Checking all n^2/2 swaps is 32k checks.
    # Each check can be done in O(n) if we precompute, or O(1) if we have the right deltas.
    
    # Let's derive the O(1) delta for swapping i and j in p.
    # Let x = p[i], y = p[j].
    # Terms involving i or j in cost:
    # sum_{k} a[i,k] b[x, p[k]] + sum_{k} a[k,i] b[p[k], x]
    # sum_{k} a[j,k] b[y, p[k]] + sum_{k} a[k,j] b[p[k], y]
    # Note: terms a[i,j]b[x,y] and a[j,i]b[y,x] are included above when k=j and k=i.
    
    # After swap, x and y are swapped in positions i and j.
    # New terms:
    # sum_{k} a[i,k] b[y, p'[k]] + sum_{k} a[k,i] b[p'[k], y]
    # where p'[i]=y, p'[j]=x, p'[k]=p[k] for k!=i,j.
    
    # This is still O(n) to compute exactly unless we precompute some vectors.
    # Let's stick to a simpler, robust approach:
    # 1. Start with rank matching.
    # 2. Perform random 2-opt swaps.
    # 3. Accept if better.
    # 4. Occasionally restart from a perturbed state or random state if stuck.
    
    # Actually, for QAP, a very effective simple heuristic is "Greedy Construction" followed by "Swap Local Search".
    # But rank matching is a good start.
    
    # Let's implement a fast swap evaluation.
    # Precompute for each facility i and location l:
    # v_i[l] = sum_{k} (a[i,k] + a[k,i]) * b[l, p[k]] ? No, b is not symmetric necessarily?
    # The task says "sum of flow times distance". Usually QAP assumes symmetric matrices, but let's check.
    # "a and b are this task's QAPLIB matrices". QAPLIB instances usually have symmetric a and b, but not always.
    # The formula is sum_{i,j} a[i,j] * b[p[i], p[j]].
    # If a and b are symmetric, we can simplify.
    # Let's assume general case for safety, but use the O(n) swap check.
    
    # O(n) swap check:
    # delta = (new_cost - old_cost)
    # Only terms involving i or j change.
    # Old contribution of i: sum_k a[i,k] b[x, p[k]] + sum_k a[k,i] b[p[k], x]
    # Old contribution of j: sum_k a[j,k] b[y, p[k]] + sum_k a[k,j] b[p[k], y]
    # Note: The term a[i,j] b[x,y] is counted in i's sum (k=j) and j's sum (k=i)?
    # In sum_{i,j}, the term for (i,j) is a[i,j]b[p[i],p[j]].
    # So yes, it's included.
    
    # To make this O(1), we can maintain an auxiliary array.
    # Let's just use the O(n) version first and see if it's fast enough.
    # 256 operations per swap check.
    # 32k swaps per full pass. 32k * 256 = 8M ops.
    # NumPy can do this in < 1ms if vectorized, but loop in Python is slow.
    # Let's try to vectorize the delta calculation for a single swap (i,j).
    
    def swap_delta(i, j, p, a, b):
        x = p[i]
        y = p[j]
        # We need to compute the change in cost.
        # It is easier to just compute the cost of the whole matrix if n is small, but n=256 is borderline.
        # Let's try the O(n) vectorized approach.
        
        # Vector of locations for all k
        locs = np.array(p)
        
        # Terms for i before:
        # a[i, :] * b[x, locs] + a[:, i] * b[locs, x]
        # Terms for j before:
        # a[j, :] * b[y, locs] + a[:, j] * b[locs, y]
        
        # Terms for i after (x becomes y at pos i, y becomes x at pos j):
        # This is tricky to vectorize without creating the new p array.
        
        # Alternative: Just compute the full cost for a few candidate swaps?
        # No, that's O(n^2) per swap.
        
        # Let's use the property:
        # Delta = sum_{k != i,j} (a[i,k] (b[y,p[k]] - b[x,p[k]]) + a[k,i] (b[p[k],y] - b[p[k],x]) +
        #                        a[j,k] (b[x,p[k]] - b[y,p[k]]) + a[k,j] (b[p[k],x] - b[p[k],y]))
        #       + (a[i,j] (b[y,x] - b[x,y]) + a[j,i] (b[x,y] - b[y,x]))
        #       + (a[i,i] (b[y,y] - b[x,x]) + a[j,j] (b[x,x] - b[y,y]))  # if i!=j, p[i]!=p[j] usually, but here x!=y
        #       # Wait, p is a permutation, so x != y.
        #       # The terms a[i,i] b[x,x] and a[j,j] b[y,y] are removed, and a[i,i] b[y,y] and a[j,j] b[x,x] are added?
        #       # No, facility i is now at y. So a[i,i] b[y,y]. Facility j is now at x. So a[j,j] b[x,x].
        #       # Old: a[i,i] b[x,x] + a[j,j] b[y,y].
        #       # New: a[i,i] b[y,y] + a[j,j] b[x,x].
        #       # Delta_self = a[i,i](b[y,y]-b[x,x]) + a[j,j](b[x,x]-b[y,y]) = (a[i,i]-a[j,j])(b[y,y]-b[x,x])
        
        # This is getting complicated to vectorize efficiently in Python without precomputation.
        
        # Let's try a different strategy: **Tabu Search** or **Simulated Annealing** with full cost recomputation.
        # Full cost recomputation is O(n^2) = 65k multiplications/additions.
        # In NumPy: `np.sum(a * b[np.ix_(p, p)])`.
        # Let's benchmark this mentally. 65k elements. ~1-2 microseconds per element? No, much faster.
        # It's likely < 100 microseconds.
        # 150 seconds / 100us = 1.5 million iterations.
        # That is plenty for a simple random swap search.
        
        pass # This function is not used in the final implementation below, just for thought process.

    # Strategy: Random Swap Local Search with Simulated Annealing.
    
    def cost_calc(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_p = np.array(best_p, dtype=np.int64)
    current_cost = best_cost
    
    # Simulated Annealing parameters
    T = 10000.0
    cooling = 0.999
    min_T = 1.0
    
    while time.time() < deadline:
        # Pick random i, j
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
            
        # Swap
        current_p[i], current_p[j] = current_p[j], current_p[i]
        new_cost = cost_calc(current_p)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = current_p.tolist()
        else:
            # Accept with probability exp(-delta/T)
            delta = new_cost - current_cost
            if T > min_T:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_cost = new_cost
                else:
                    # Revert
                    current_p[i], current_p[j] = current_p[j], current_p[i]
            else:
                # Revert
                current_p[i], current_p[j] = current_p[j], current_p[i]
        
        T *= cooling
        
        # If stuck, maybe restart?
        # For now, just continue.
        
    return best_p
# EVOLVE-BLOCK-END
