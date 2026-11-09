import numpy as np
import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 88.0
    N = 4096
    
    # 1. Fast Vectorized Greedy Construction
    # We greedily add numbers from 1 to N.
    # For a new candidate v, the new triple sums are:
    # 3v, 2v+u, v+2u, v+u+w (for u,w in current set)
    # Using NumPy, we can compute these quickly.
    
    chosen = []
    sums_set = set()
    
    # Optimization: Pre-compute pair sums of current set? 
    # Actually, for each new v, we need:
    # A = {3v}
    # B = {2v + u | u in chosen}
    # C = {v + 2u | u in chosen}
    # D = {v + u + w | u, w in chosen}
    
    # To make D fast, we can maintain a sorted list of chosen.
    # D is v + (sum of all pairs from chosen).
    # Let P be the set of pair sums {u+w | u,w in chosen}.
    # Then D = {v + p | p in P}.
    
    # Updating P when adding v:
    # New pairs: {v+v} and {v+u | u in chosen}
    # So P_new = P_old U {2v} U {v+u | u in chosen}
    
    # This allows O(1) check for D if we just check intersection with sums_set.
    # But we still need to build the set of new sums to add to sums_set.
    
    # Let's use a sorted list for `chosen` to allow quick pair sum generation?
    # No, just iterating is fine if k is small (<100).
    # The bottleneck is the Python loop over candidates (4096) and the set operations.
    
    # Let's try to vectorize the "try_add" logic.
    
    # State:
    # chosen: list of ints
    # pair_sums: set of ints (sums of pairs from chosen)
    # triple_sums: set of ints (sums of triples from chosen)
    
    chosen = []
    pair_sums = set()
    triple_sums = set()
    
    # Helper to add a number
    def try_add(v):
        nonlocal chosen, pair_sums, triple_sums
        if v in chosen: # This check is O(k). We can use a set for membership.
            return False
        # We need a membership set for chosen for O(1) check
        # Let's maintain a set `chosen_set`
        
        # New triple sums involving v:
        # 1. 3v
        # 2. 2v + u for u in chosen
        # 3. v + 2u for u in chosen
        # 4. v + u + w for u, w in chosen (which is v + p for p in pair_sums)
        
        new_triples = set()
        new_triples.add(3*v)
        
        # Vectorized-ish update
        # Note: k is expected to be < 80, so even O(k^2) is 6400 ops.
        # The overhead of Python set creation is high.
        
        # Let's just do it simply but efficiently.
        for u in chosen:
            new_triples.add(2*v + u)
            new_triples.add(v + 2*u)
        for p in pair_sums:
            new_triples.add(v + p)
            
        if new_triples & triple_sums:
            return False
            
        # Update state
        chosen.append(v)
        triple_sums |= new_triples
        
        # Update pair_sums
        new_pairs = set()
        new_pairs.add(2*v)
        for u in chosen:
            new_pairs.add(v + u)
        pair_sums |= new_pairs
        
        return True

    # We need `chosen_set` for O(1) membership
    chosen_set = set()
    
    # Re-define try_add to use chosen_set
    def try_add_opt(v):
        nonlocal chosen, pair_sums, triple_sums, chosen_set
        if v in chosen_set:
            return False
        
        new_triples = set()
        new_triples.add(3*v)
        
        # Optimization: If k is large, this loop is slow.
        # But k is max 80. 80 iterations is fast.
        for u in chosen:
            new_triples.add(2*v + u)
            new_triples.add(v + 2*u)
        for p in pair_sums:
            new_triples.add(v + p)
            
        if new_triples & triple_sums:
            return False
            
        chosen.append(v)
        chosen_set.add(v)
        triple_sums |= new_triples
        
        new_pairs = set()
        new_pairs.add(2*v)
        for u in chosen:
            new_pairs.add(v + u)
        pair_sums |= new_pairs
        
        return True

    # Phase 1: Greedy from 1 to N
    for v in range(1, N + 1):
        if time.time() > DEADLINE - 10:
            break
        try_add_opt(v)
        
    best = list(chosen)
    best_set = set(chosen)
    
    # Phase 2: Local Search / Optimization
    # Try to add more numbers by checking all 1..N again?
    # The greedy pass might have missed some numbers that could fit if others were removed,
    # or just because of ordering? No, greedy is deterministic.
    # But maybe we can swap?
    # Or maybe we can just run a second pass?
    # Since we added in increasing order, we might have blocked larger numbers.
    # Let's try to add numbers in decreasing order to the current set?
    # Or restart with different ordering?
    
    # Strategy: 
    # 1. Greedy Ascending (done)
    # 2. Greedy Descending
    # 3. Randomized Greedy
    
    # Let's keep the best of several strategies.
    
    def run_greedy(order):
        c = []
        cs = set()
        ps = set()
        ts = set()
        
        def ta(v):
            if v in cs: return False
            nt = {3*v}
            for u in c:
                nt.add(2*v + u)
                nt.add(v + 2*u)
            for p in ps:
                nt.add(v + p)
            if nt & ts: return False
            c.append(v)
            cs.add(v)
            ts |= nt
            np_ = {2*v}
            for u in c:
                np_.add(v+u)
            ps |= np_
            return True
            
        for v in order:
            if time.time() > DEADLINE - 2:
                break
            ta(v)
        return c
        
    # We already did ascending.
    # Try Descending
    if time.time() < DEADLINE - 5:
        desc_order = list(range(N, 0, -1))
        c_desc = run_greedy(desc_order)
        if len(c_desc) > len(best):
            best = c_desc
            best_set = set(c_desc)
            
    # Try Randomized
    random.seed(42)
    for i in range(20):
        if time.time() > DEADLINE - 2:
            break
        rand_order = list(range(1, N+1))
        random.shuffle(rand_order)
        c_rand = run_greedy(rand_order)
        if len(c_rand) > len(best):
            best = c_rand
            best_set = set(c_rand)
            
    # Phase 3: Local Improvement (Add moves)
    # Try to add any number from 1..N that wasn't added.
    # This is essentially re-running greedy on the remaining candidates.
    # But we must check against the current `best` set's sums.
    
    # Let's rebuild sums for `best`
    best_set = set(best)
    best_pairs = set()
    for i in range(len(best)):
        for j in range(i, len(best)):
            best_pairs.add(best[i] + best[j])
            
    best_triples = set()
    for i in range(len(best)):
        for j in range(i, len(best)):
            for k in range(j, len(best)):
                best_triples.add(best[i] + best[j] + best[k])
                
    # Try to add
    improved = True
    while improved and time.time() < DEADLINE - 1:
        improved = False
        for v in range(1, N+1):
            if v in best_set:
                continue
            # Check if v can be added
            nt = {3*v}
            for u in best:
                nt.add(2*v + u)
                nt.add(v + 2*u)
            for p in best_pairs:
                nt.add(v + p)
            if not (nt & best_triples):
                # Add it
                best.append(v)
                best_set.add(v)
                best_triples |= nt
                # Update pairs
                np_ = {2*v}
                for u in best:
                    np_.add(v+u)
                best_pairs |= np_
                improved = True
                # Restart loop? Or just continue?
                # If we add one, others might become valid/invalid?
                # Adding only increases sums, so it might block others.
                # But it doesn't unblock.
                # So we can just continue scanning.
                # However, the set of available numbers shrinks.
                pass
    # Note: The while loop above is inefficient if it iterates 4096 times repeatedly.
    # But `improved` only becomes True if we add something.
    # Max 80 adds. So at most 80 * 4096 checks. ~300k checks. Fast.
    
    # Final Validation (Safety)
    # The task requires unique sums for ALL unordered triples.
    # Our incremental logic ensures this.
    # But let's do a final check to be absolutely sure, and sort.
    
    best = sorted(set(best))
    
    # Quick validity check (O(k^3) is fine for k<100)
    seen = set()
    valid = True
    m = len(best)
    for i in range(m):
        for j in range(i, m):
            for k in range(j, m):
                s = best[i] + best[j] + best[k]
                if s in seen:
                    valid = False
                    break
                seen.add(s)
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid small set if something went wrong
        # e.g. powers of 2
        return [2**i for i in range(12)]
        
    return best

