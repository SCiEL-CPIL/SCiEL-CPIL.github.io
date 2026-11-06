import numpy as np
import time, random, itertools, sys

def construct():
    t0 = time.time()
    DEADLINE = 110.0   # safety margin before 150s limit
    
    n = 16
    # Precompute popcount for all subsets
    pop = np.zeros(1<<n, dtype=np.int8)
    for i in range(1, 1<<n):
        pop[i] = pop[i>>1] + (i&1)
    
    # Precompute all triples and their properties
    all_triples = list(itertools.combinations(range(n), 3))
    triple_masks = [ (1<<a)|(1<<b)|(1<<c) for a,b,c in all_triples ]
    num_triples = len(all_triples)
    
    # Pair mask for linearity check
    def pair_mask(a,b,c):
        pm = 0
        for x,y in ((a,b),(a,c),(b,c)):
            if x>y: x,y=y,x
            pm |= 1 << (x*n + y)
        return pm
    tpair = [pair_mask(*t) for t in all_triples]
    
    # Fast independence number calculation
    def independence_fast(triple_mask_list):
        """Compute independence number given list of triple masks."""
        if not triple_mask_list:
            return n
        # Build covered array using bit operations
        covered = np.zeros(1<<n, dtype=bool)
        for tm in triple_mask_list:
            # All subsets of tm are covered
            sub = tm
            while sub:
                covered[sub] = True
                sub = (sub-1) & tm
            covered[0] = True
        if covered.all():
            return 0
        return int(pop[~covered].max())
    
    # Known construction: projective plane of order 3 (13 points, 13 lines)
    # Extended to 16 points
    # PG(2,3) has 13 points and 13 lines, independence number 4
    # We'll construct a system on 16 points with small independence number
    
    # PG(2,3) construction on points 0-12
    pg23_lines = [
        [0,1,2], [3,4,5], [6,7,8], [9,10,11],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [1,5,6], [2,3,7],
        [9,12,13], [10,12,14], [11,12,15]
    ]
    # Wait, that's not right. Let me use the standard PG(2,3)
    # Points: (x,y,z) over GF(3) up to scale, 13 points
    # Let me use a known STS(13) - the projective plane of order 3
    
    # Standard STS(13) from projective plane PG(2,3)
    # Using 0-based indexing
    st_s13 = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [1, 5, 6], [2, 3, 7],
        [9, 12, 13], [10, 12, 14], [11, 12, 15]
    ]
    # This is not quite right either. Let me use the actual PG(2,3)
    
    # Proper PG(2,3) construction:
    # Points represented as [x:y:z] over GF(3), not all zero, up to scalar
    # There are (3^3 - 1)/(3-1) = 13 points
    # Lines: each line has 4 points (order 3 + 1)
    # Wait, PG(2,3) has order 3, so each line has 3+1=4 points? No.
    # PG(2,q) has q^2+q+1 points and lines, each line has q+1 points
    # For q=3: 13 points, 13 lines, each line has 4 points
    # But we want triples (3 points per line), so this is a 4-uniform hypergraph
    
    # For our task, we need triples. Let me use a different approach.
    # Use a Steiner triple system S(2,3,7) - Fano plane on 7 points, 
    # and extend to 16 points.
    
    # Actually, let me just do a good local search with better initialization.
    
    random.seed(42)
    
    # Greedy packing
    def greedy_packing(order):
        sys_set = set()
        pair_union = 0
        for idx in order:
            if (tpair[idx] & pair_union) == 0:
                sys_set.add(idx)
                pair_union |= tpair[idx]
        return sys_set, pair_union
    
    best_sys = None
    best_indep = n + 1
    
    # Multiple greedy starts
    for attempt in range(50):
        if time.time() - t0 > DEADLINE:
            break
        order = list(range(num_triples))
        random.shuffle(order)
        sys_set, pair_union = greedy_packing(order)
        cur_indep = independence_fast([triple_masks[i] for i in sys_set])
        if cur_indep < best_indep:
            best_indep = cur_indep
            best_sys = list(sys_set)
    
    # Local search to improve
    if best_sys is not None:
        sys_set = set(best_sys)
        pair_union = 0
        for i in sys_set:
            pair_union |= tpair[i]
        cur_indep = best_indep
        
        # Local search with restarts
        no_improve_count = 0
        while time.time() - t0 < DEADLINE:
            r = random.random()
            if r < 0.4:
                # Add a random valid triple
                cand = random.randrange(num_triples)
                if (tpair[cand] & pair_union) == 0 and cand not in sys_set:
                    sys_set.add(cand)
                    pair_union |= tpair[cand]
                    ni = independence_fast([triple_masks[i] for i in sys_set])
                    if ni <= cur_indep:
                        cur_indep = ni
                        no_improve_count = 0
                        if cur_indep < best_indep:
                            best_indep = cur_indep
                            best_sys = list(sys_set)
                    else:
                        sys_set.discard(cand)
                        pair_union &= ~tpair[cand]
                        no_improve_count += 1
            elif r < 0.7:
                # Swap: remove one, add another
                if not sys_set:
                    continue
                rem = random.choice(list(sys_set))
                cand = random.randrange(num_triples)
                if cand in sys_set or cand == rem:
                    continue
                new_set = set(sys_set)
                new_set.discard(rem)
                new_set.add(cand)
                # Check linearity
                ok = True
                pm = 0
                for i in new_set:
                    if (tpair[i] & pm) != 0:
                        ok = False
                        break
                    pm |= tpair[i]
                if not ok:
                    continue
                ni = independence_fast([triple_masks[i] for i in new_set])
                if ni <= cur_indep:
                    sys_set = new_set
                    pair_union = pm
                    cur_indep = ni
                    no_improve_count = 0
                    if cur_indep < best_indep:
                        best_indep = cur_indep
                        best_sys = list(sys_set)
                else:
                    no_improve_count += 1
            else:
                # Remove one
                if len(sys_set) > 5:
                    rem = random.choice(list(sys_set))
                    new_set = set(sys_set)
                    new_set.discard(rem)
                    pm = 0
                    for i in new_set:
                        pm |= tpair[i]
                    ni = independence_fast([triple_masks[i] for i in new_set])
                    if ni <= cur_indep:
                        sys_set = new_set
                        pair_union = pm
                        cur_indep = ni
                        no_improve_count = 0
                        if cur_indep < best_indep:
                            best_indep = cur_indep
                            best_sys = list(sys_set)
                    else:
                        no_improve_count += 1
            
            # Restart if stuck
            if no_improve_count > 200:
                no_improve_count = 0
                # Restart from best
                if best_sys is not None:
                    sys_set = set(best_sys)
                    pair_union = 0
                    for i in sys_set:
                        pair_union |= tpair[i]
                    cur_indep = best_indep
                else:
                    order = list(range(num_triples))
                    random.shuffle(order)
                    sys_set, pair_union = greedy_packing(order)
                    cur_indep = independence_fast([triple_masks[i] for i in sys_set])
                    if cur_indep < best_indep:
                        best_indep = cur_indep
                        best_sys = list(sys_set)
    
    # Fallback: return a valid system
    if best_sys is None:
        # Simple greedy
        order = list(range(num_triples))
        random.shuffle(order)
        best_sys, _ = greedy_packing(order)
    
    out = [list(all_triples[i]) for i in sorted(best_sys)]
    return out
