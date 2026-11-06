import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START

def _oet_network():
    """Generate valid Odd-Even Transposition Network (120 comparators)."""
    net = []
    for phase in range(N):
        if phase % 2 == 0:
            # Even phase: (0,1), (2,3), ..., (N-2, N-1)
            for i in range(0, N - 1, 2):
                net.append((i, i + 1))
        else:
            # Odd phase: (1,2), (3,4), ..., (N-3, N-2)
            # i+1 starts at 1, i+2 ends at N-1
            # i ranges 0, 2, ..., N-3
            for i in range(0, N - 2, 2):
                net.append((i + 1, i + 2))
    return net

def _is_sorted_network(net, inputs):
    """Check if network sorts all inputs. inputs is (65536, 16) array."""
    wires = inputs.copy()
    for i, j in net:
        lo = np.minimum(wires[i], wires[j])
        hi = np.maximum(wires[i], wires[j])
        wires[i] = lo
        wires[j] = hi
    return bool(np.all(np.diff(wires, axis=0) >= 0))

def _local_search(pool, inputs, deadline, start_time=None):
    if start_time is None:
        start_time = time.time()
    
    # Ensure pool is valid and sorted
    if not _is_sorted_network(pool, inputs):
        return None

    best = list(pool)
    best_count = len(best)
    rng = random.Random(555)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Greedy Removal
        idxs = list(range(len(best)))
        rng.shuffle(idxs)
        for idx in idxs:
            if time.time() >= deadline:
                break
            if idx >= len(best):
                continue
            cand = best[:idx] + best[idx + 1:]
            if _is_sorted_network(cand, inputs):
                best = cand
                best_count = len(best)
                improved = True
                break # Restart removal loop with new shorter list
                
        if improved:
            continue
            
        # 2. Random Replacement (Swap one comparator for another)
        # Try to replace a random comparator with a random unused one
        tried = 0
        while tried < 500 and time.time() < deadline:
            tried += 1
            if len(best) == 0:
                break
            idx = rng.randrange(len(best))
            i, j = sorted(rng.sample(range(N), 2))
            if (i, j) in best:
                continue
            cand = best[:idx] + [(i, j)] + best[idx + 1:]
            if _is_sorted_network(cand, inputs):
                best = cand
                improved = True
                break
        
        if improved:
            continue

        # 3. Random Swap (Reorder) - doesn't change count, but might help if we add logic later? 
        # For now, reordering doesn't help minimization directly unless we enable 2-removals which depend on order?
        # Actually, order matters for "removal" success? No, removal just checks if the set minus one element sorts.
        # So reordering the list doesn't change the set. It's useless for minimization.
        # However, if we were doing depth optimization, it would matter.
        
        # 4. Restart with new pool if stuck
        if time.time() < deadline - 2.0:
             # Generate a new random pool based on OET + randoms
             base = _oet_network()
             random.shuffle(base)
             new_pool = list(base)
             seen = set(base)
             # Add some random comparators to explore
             while len(new_pool) < len(base) + 10:
                 i, j = sorted(rng.sample(range(N), 2))
                 if (i, j) not in seen:
                     seen.add((i, j))
                     new_pool.append((i, j))
             random.shuffle(new_pool)
             
             if _is_sorted_network(new_pool, inputs):
                 best = new_pool
                 best_count = len(best)
                 continue
                 
        # If no improvement and no restart possible, break
        break
        
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin for 90s limit
    
    # Pre-compute all zero-one inputs
    inputs = np.array(np.unravel_index(np.arange(2 ** N), (2,) * N)).T
    
    # Initial best: OET (120 comparators)
    best = _oet_network()
    
    # Local Search Loop
    # We will try multiple restarts with different random seeds/pools
    for trial in range(10):
        if time.time() >= deadline - 5.0:
            break
            
        # Create a candidate pool: OET + Random Comparators
        base = _oet_network()
        random.seed(1000 + trial)
        random.shuffle(base)
        
        pool = list(base)
        seen = set(base)
        
        # Add random comparators to allow escape from local minima
        target_pool_size = 130
        while len(pool) < target_pool_size:
            i, j = sorted(random.sample(range(N), 2))
            if (i, j) not in seen:
                seen.add((i, j))
                pool.append((i, j))
        
        random.shuffle(pool)
        
        # Verify pool is valid (it should be, as it's a superset of OET)
        if not _is_sorted_network(pool, inputs):
            continue
            
        # Run local search on this pool
        candidate = _local_search(pool, inputs, deadline, start_time=start_time)
        
        if candidate is not None and _is_sorted_network(candidate, inputs):
            if len(candidate) < len(best):
                best = candidate
                
    # Final check
    if not _is_sorted_network(best, inputs):
        # Should not happen, but return OET if something went wrong
        best = _oet_network()
        
    return best

# EVOLVE-BLOCK-END
