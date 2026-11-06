import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # Safety margin
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2)s
    # A K(2,2,2) is defined by 3 disjoint pairs (x1,x2), (y1,y2), (z1,z2).
    # The 8 transversals are (xi, yj, zk) for i,j,k in {1,2}.
    # We identify each K(2,2,2) by its 6 vertices.
    # There are C(16,6) * 15 partitions of 6 items into 3 pairs.
    # C(16,6) = 8008. 15 partitions. Total ~120k? 
    # Actually, let's just iterate.
    
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    # Generate all 6-subsets
    # To be fast, we can iterate through combinations.
    for six_subset in combinations(range(n), 6):
        v = list(six_subset)
        # Generate all perfect matchings of 6 vertices into 3 pairs.
        # There are 15 such matchings.
        # We can hardcode the 15 partitions of indices 0..5.
        # Indices: 0,1,2,3,4,5
        # Pairs: (0,1),(2,3),(4,5)
        # (0,1),(2,4),(3,5)
        # ...
        
        # Efficient way to generate 15 matchings:
        # Fix 0 with 1,2,3,4,5 (5 choices).
        # Then fix next smallest with remaining 3 (3 choices).
        # Last 2 are fixed.
        # Total 5*3 = 15.
        
        for i in range(1, 6):
            p1 = (v[0], v[i])
            rem = [v[j] for j in range(6) if j != 0 and j != i]
            # rem has 4 elements. Fix rem[0] with rem[1], rem[2], rem[3]
            for j in range(1, 4):
                p2 = (rem[0], rem[j])
                # remaining 2 elements for p3
                if j == 1:
                    p3 = (rem[2], rem[3])
                elif j == 2:
                    p3 = (rem[1], rem[3])
                else:
                    p3 = (rem[1], rem[2])
                
                # Construct the 8 triples
                t_ids = []
                for x in p1:
                    for y in p2:
                        for z in p3:
                            t = tuple(sorted((x, y, z)))
                            t_id = triple_to_id[t]
                            t_ids.append(t_id)
                
                k_idx = len(k222_triples)
                k222_triples.append(t_ids)
                
                for t_id in t_ids:
                    triple_to_k222s[t_id].append(k_idx)
                    
    num_k222s = len(k222_triples)
    
    # Local Search Solver
    # State:
    # active: boolean array of size num_triples
    # k_count: array of size num_k222s, count of active triples in each K
    # We want to maximize sum(active) such that max(k_count) < 8.
    
    def solve_instance(seed):
        rng = random.Random(seed)
        active = [False] * num_triples
        k_count = [0] * num_k222s
        
        # Greedy Initialization
        # Random order
        order = list(range(num_triples))
        rng.shuffle(order)
        
        for t_id in order:
            # Check if adding t_id violates any constraint
            # A constraint is violated if any k containing t_id has count 7
            ok = True
            for k_idx in triple_to_k222s[t_id]:
                if k_count[k_idx] >= 7:
                    ok = False
                    break
            if ok:
                active[t_id] = True
                for k_idx in triple_to_k222s[t_id]:
                    k_count[k_idx] += 1
        
        current_size = sum(active)
        
        # Local Search: Try to add triples. If stuck, swap.
        # To speed up, we maintain a list of "addable" triples?
        # Or just pick random candidates.
        
        # Optimization: Precompute for each triple, the max k_count among its Ks.
        # If max_k_count < 7, it is addable.
        
        # Let's do a loop of random moves.
        # Move types:
        # 1. Add a triple t if valid. (Increases size)
        # 2. Swap t_out with t_in if valid. (Maintains size, might allow future adds)
        
        # We will run this for a fixed number of steps or time.
        steps = 5000
        for step in range(steps):
            if time.time() > DEADLINE:
                break
            
            # Try to add
            # Pick a random inactive triple
            # To be faster, we can pick from a precomputed list of inactive triples?
            # Or just random index.
            t_in = rng.randint(0, num_triples - 1)
            if not active[t_in]:
                # Check validity
                can_add = True
                for k_idx in triple_to_k222s[t_in]:
                    if k_count[k_idx] >= 7:
                        can_add = False
                        break
                if can_add:
                    active[t_in] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[t_in]:
                        k_count[k_idx] += 1
                    continue
            
            # If cannot add (or was already active), try a swap
            # Pick t_out from active
            # Pick t_in from inactive
            # To avoid picking same, just pick two randoms and check.
            
            # Efficient swap:
            # Pick t_out randomly from active.
            # Pick t_in randomly from inactive.
            
            # To pick from active quickly, we can just try random indices until we find one.
            # Since density might be high, this is fast.
            
            t_out = rng.randint(0, num_triples - 1)
            if not active[t_out]:
                continue
                
            t_in = rng.randint(0, num_triples - 1)
            if active[t_in]:
                continue
            
            # Check if swap is valid.
            # Remove t_out, Add t_in.
            # Valid if adding t_in doesn't cause any K to reach 8, considering t_out is removed.
            # For each K containing t_in:
            #   new_count = k_count[k] - (1 if t_out in K else 0) + 1
            #   if new_count >= 8: invalid
            
            # Precompute: does t_out share Ks with t_in?
            # We can just iterate Ks of t_in.
            
            valid_swap = True
            for k_idx in triple_to_k222s[t_in]:
                # Calculate count if t_out removed and t_in added
                # k_count[k_idx] currently includes t_out if t_out is in this K.
                # We need to know if t_out is in k222_triples[k_idx].
                # This check is O(1) if we had a set, but k222_triples is a list of 8 ints.
                # 8 is small.
                
                # Check if t_out is in this K
                t_out_in_k = False
                for t in k222_triples[k_idx]:
                    if t == t_out:
                        t_out_in_k = True
                        break
                
                new_count = k_count[k_idx] - (1 if t_out_in_k else 0) + 1
                if new_count >= 8:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Execute swap
                active[t_out] = False
                for k_idx in triple_to_k222s[t_out]:
                    k_count[k_idx] -= 1
                
                active[t_in] = True
                for k_idx in triple_to_k222s[t_in]:
                    k_count[k_idx] += 1
                # current_size unchanged
        
        return current_size, active

    best_size = 0
    best_active = None
    
    # Run multiple instances
    # Estimate time per instance.
    # 5000 steps * ~100 ops/step = 500k ops. Very fast.
    # We can run many more steps or more instances.
    
    # Let's run until deadline.
    seed = 0
    while time.time() < DEADLINE:
        size, active = solve_instance(seed)
        if size > best_size:
            best_size = size
            best_active = active[:]
        seed += 1
        
        # Optional: Polish the best solution found so far?
        # The local search already does greedy init + random walks.
        # We can try to do a "greedy fill" on the best solution to see if we missed easy adds.
        if seed % 5 == 0 and best_active:
             # Quick polish: try adding any triple that fits
             for t_id in range(num_triples):
                 if not best_active[t_id]:
                     ok = True
                     # Need k_count for best_active to check this efficiently?
                     # We didn't keep k_count for best_active.
                     # Reconstructing k_count for 560 triples is fast.
                     pass # Skip for now to keep code simple and fast. The local search is strong.
             
    if best_active is None:
        return []
        
    return [list(triples[i]) for i in range(num_triples) if best_active[i]]
