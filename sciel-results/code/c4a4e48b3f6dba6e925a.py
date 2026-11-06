import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Precompute masks for all triples
    triple_masks = [0] * num_triples
    for i, t in enumerate(all_triples):
        triple_masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        
    # Precompute pair masks for linearity check
    tpair = [0] * num_triples
    for i, (a, b, c) in enumerate(all_triples):
        pm = 0
        for x, y in ((a, b), (a, c), (b, c)):
            if x > y:
                x, y = y, x
            pm |= 1 << (x * n + y)
        tpair[i] = pm
        
    # Precompute popcount array for all 2^16 masks
    pop_arr = np.zeros(1 << n, dtype=np.int8)
    for i in range(1, 1 << n):
        pop_arr[i] = pop_arr[i >> 1] + (i & 1)
        
    # Precompute subsets array for vectorized coverage check
    subsets = np.arange(1 << n, dtype=np.int32)
    
    def compute_alpha_fast(triple_indices):
        """
        Compute the independence number of the hypergraph defined by triple_indices.
        Returns the size of the largest subset of points containing no triple.
        Uses vectorized NumPy operations for speed.
        """
        if not triple_indices:
            return n
        
        covered = np.zeros(1 << n, dtype=bool)
        
        for idx in triple_indices:
            t_mask = triple_masks[idx]
            covered |= (subsets & t_mask) == t_mask
            
        max_alpha = np.max(pop_arr[~covered])
        return int(max_alpha)
    
    def greedy_packing(order):
        """
        Greedily add triples in the given order if they are linear with existing ones.
        """
        sys_set = set()
        pair_union = 0
        for idx in order:
            if (tpair[idx] & pair_union) == 0:
                sys_set.add(idx)
                pair_union |= tpair[idx]
        return list(sys_set), pair_union
    
    # Initial solution
    order = list(range(num_triples))
    random.shuffle(order)
    best_packing, _ = greedy_packing(order)
    best_alpha = compute_alpha_fast(best_packing)
    
    # Local Search
    current_packing = list(best_packing)
    current_pair_union = 0
    for idx in current_packing:
        current_pair_union |= tpair[idx]
    current_alpha = best_alpha
    current_set = set(current_packing)
    
    # Track best alpha found
    global_best_alpha = best_alpha
    global_best_packing = list(best_packing)
    
    # Precompute list of all triples for random selection
    # We already have all_triples and indices 0..559
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.5:
            # Try to add a random valid triple
            cand = random.randrange(num_triples)
            if cand not in current_set and (tpair[cand] & current_pair_union) == 0:
                new_packing = current_packing + [cand]
                new_alpha = compute_alpha_fast(new_packing)
                if new_alpha <= current_alpha:
                    current_packing = new_packing
                    current_set.add(cand)
                    current_pair_union |= tpair[cand]
                    current_alpha = new_alpha
                    if current_alpha < global_best_alpha:
                        global_best_alpha = current_alpha
                        global_best_packing = list(current_packing)
                        
        elif r < 0.9:
            # Swap: remove one, add one
            if not current_packing:
                continue
            rem = random.choice(current_packing)
            cand = random.randrange(num_triples)
            if cand in current_set or cand == rem:
                continue
            
            # Check linearity of new set: current_set - {rem} + {cand}
            conflict = False
            for other in current_set:
                if other == rem:
                    continue
                if (tpair[cand] & tpair[other]) != 0:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            new_packing = [x for x in current_packing if x != rem] + [cand]
            new_alpha = compute_alpha_fast(new_packing)
            
            if new_alpha <= current_alpha:
                current_packing = new_packing
                current_set.discard(rem)
                current_set.add(cand)
                # Recompute pair_union
                current_pair_union = 0
                for idx in current_packing:
                    current_pair_union |= tpair[idx]
                current_alpha = new_alpha
                if current_alpha < global_best_alpha:
                    global_best_alpha = current_alpha
                    global_best_packing = list(current_packing)
                    
        else:
            # Perturbation: Remove 1-3 random triples to escape local minima
            # Then greedily refill
            n_remove = random.randint(1, 3)
            for _ in range(n_remove):
                if current_packing:
                    rem = random.choice(current_packing)
                    current_packing.remove(rem)
                    current_set.discard(rem)
            
            # Recompute pair_union
            current_pair_union = 0
            for idx in current_packing:
                current_pair_union |= tpair[idx]
                
            # Greedily add back valid triples
            candidates = list(range(num_triples))
            random.shuffle(candidates)
            for cand in candidates:
                if cand not in current_set and (tpair[cand] & current_pair_union) == 0:
                    new_packing = current_packing + [cand]
                    new_alpha = compute_alpha_fast(new_packing)
                    if new_alpha <= current_alpha:
                        current_packing = new_packing
                        current_set.add(cand)
                        current_pair_union |= tpair[cand]
                        current_alpha = new_alpha
            
            # After perturbation and refill, update best
            if current_alpha < global_best_alpha:
                global_best_alpha = current_alpha
                global_best_packing = list(current_packing)
    
    # Convert best_packing (list of indices) to list of triples
    out = [list(all_triples[i]) for i in global_best_packing]
    return out

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Num triples: {len(res)}")
    # Verify linearity
    masks = set()
    for t in res:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        if m in masks:
            print("Duplicate triple found!")
        masks.add(m)
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            mi = (1 << res[i][0]) | (1 << res[i][1]) | (1 << res[i][2])
            mj = (1 << res[j][0]) | (1 << res[j][1]) | (1 << res[j][2])
            inter = mi & mj
            count = bin(inter).count('1')
            if count > 1:
                print(f"Non-linear: {res[i]} and {res[j]} share {count} points")
    
    # Compute alpha
    def check_alpha(triples):
        n = 16
        pop_arr = np.zeros(1 << n, dtype=np.int8)
        for i in range(1, 1 << n):
            pop_arr[i] = pop_arr[i >> 1] + (i & 1)
        covered = np.zeros(1 << n, dtype=bool)
        subsets = np.arange(1 << n, dtype=np.int32)
        for t in triples:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            covered |= (subsets & m) == m
        return int(np.max(pop_arr[~covered]))
    
    print(f"Alpha: {check_alpha(res)}")
