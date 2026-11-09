import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    n = 16
    k = 4
    max_target = 250

    # 1. Generate all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    total = len(all_sets)
    
    # Map mask to index for O(1) lookup
    mask_to_idx = {m: i for i, m in enumerate(all_sets)}

    # 2. Precompute InterSets
    # InterSets[i][I] = list of indices j such that all_sets[j] & all_sets[i] == I
    # I must be a subset of all_sets[i]
    InterSets = [{} for _ in range(total)]
    
    # To speed up, we can iterate over all pairs? No, that's 2184^2.
    # Better: For each i, for each subset I of all_sets[i], find j.
    # But finding j such that j & i == I is still O(N) per I.
    # Total work: Sum over i of (2^k * N) = 16 * 2184 * 2184 ~ 76M ops. Might be too slow for Python.
    
    # Alternative Precomputation Strategy:
    # We only need InterSets for the pairs that actually appear in F during search.
    # But F changes.
    
    # Let's try a different update strategy that doesn't require precomputed InterSets for ALL pairs.
    # Instead, when we need to update counts for a new pair (u, m), we scan all s in AllSets?
    # Cost: 2184 checks per pair.
    # Pairs added per step: |F| ~ 250.
    # Cost per step: 250 * 2184 ~ 546,000 ops.
    # If we do 1000 steps, that's 500M ops. Too slow for Python in 60s.
    
    # We need the precomputed InterSets, but we must build them efficiently.
    # Optimization:
    # For a fixed i, the condition s & i == I implies s must contain I and s must be disjoint from i \ I.
    # We can generate all s that contain I and are disjoint from i \ I.
    # Number of such s is binomial(n - |I| - |i \ I|, k - |I|) = binomial(n - |i|, k - |I|).
    # Since |i|=4, max is binomial(12, 4) = 495.
    # So for each i, for each I subset of i, we can generate the list of candidates directly.
    # Total candidates to generate: Sum_i Sum_{I subset i} binom(12, 4-|I|).
    # For one i: 
    # |I|=0: binom(12,4)=495
    # |I|=1: 4 * binom(12,3)=4*220=880
    # |I|=2: 6 * binom(12,2)=6*66=396
    # |I|=3: 4 * binom(12,1)=4*12=48
    # |I|=4: 1 * binom(12,0)=1
    # Total per i: 495+880+396+48+1 = 1820.
    # Total for all i: 2184 * 1820 ~ 4M operations. This is very fast in Python.
    
    # Let's implement this generation.
    
    # Helper to get all masks of size k containing a specific mask 'core' and disjoint from 'forbidden'
    # We can just iterate all_sets and filter? 2184 * 1820 checks = 4M checks. 
    # "Check" is (s & i) == I.
    # 4M bitwise ops is fine.
    
    for i, mask_i in enumerate(all_sets):
        # Get all subsets of mask_i
        sub = mask_i
        while True:
            # Find all j such that all_sets[j] & mask_i == sub
            # Optimization: Pre-filter?
            # Just loop all j? 2184 * 16 * 2184 = 76M. 
            # 76M simple ops might take ~5-10 seconds in Python. 
            # Let's try to optimize.
            
            # We can store InterSets[i][sub] as a list of indices.
            # To speed up, we can iterate j only if (all_sets[j] & mask_i) is a subset of mask_i? 
            # No, (j & i) is always subset of i.
            
            # Let's just do the O(N^2 * 2^k) precomputation but carefully.
            # Actually, we can do it in O(N * 2^k * AvgCandidates) if we generate candidates.
            
            # Let's stick to the O(N^2 * 2^k) but see if we can vectorize or simplify.
            # 76M iterations in Python is likely too slow (approx 5-10s).
            # We have 55s budget. It's risky but possible.
            
            # Better: Generate candidates combinatorially.
            # For fixed i and sub I:
            # We need s such that I subset s and (s & (mask_i ^ I)) == 0.
            # Let remaining = n - |I| - |mask_i ^ I| = n - |mask_i| = 12.
            # We need to choose k - |I| elements from the 12 available.
            # We can precompute all 4-subsets.
            # For each I, generate all supersets of size 4 from the 12 available elements.
            
            pass # Placeholder for logic
        
        if sub == 0: break
        sub = (sub - 1) & mask_i

    # This precomputation is getting complex. Let's use the "Scan All" method but optimize the inner loop.
    # Or, better: Don't precompute InterSets for ALL i. 
    # Only compute when needed? 
    # If we compute on the fly, cost is 2184 per pair.
    # 250 pairs * 2184 = 546k ops per step.
    # If we do 100 steps, 54M ops. ~5s. 
    # This is acceptable! 
    # Let's NOT precompute InterSets globally. 
    # Instead, maintain `conflict_count` and update it by scanning `all_sets` when adding/removing.
    
    # State:
    # F: list of masks in family
    # InF: set of masks in F
    # conflict_count: dict mask -> count (for masks NOT in F)
    
    # Initialize with Greedy
    F = []
    InF = set()
    conflict_count = {}
    
    # Initialize conflict_count to 0 for all
    for m in all_sets:
        conflict_count[m] = 0
        
    # Greedy Initialization
    # Use a good order. Random shuffle is fine.
    rng = random.Random(42)
    order = list(range(total))
    rng.shuffle(order)
    
    for idx in order:
        m = all_sets[idx]
        if conflict_count[m] == 0:
            # Add m
            F.append(m)
            InF.add(m)
            
            # Update conflict_count for all other s
            # For each u in F (excluding m), pair (u, m) creates conflicts
            for u in F[:-1]:
                I = u & m
                # Find all s not in F such that s&u == I and s&m == I
                # Scan all_sets
                for s in all_sets:
                    if s in InF: continue
                    if (s & u) == I and (s & m) == I:
                        conflict_count[s] += 1
                        
    best_size = len(F)
    best_F = F[:]
    
    # Local Search
    # We need to be able to "undo" a removal.
    # We will store the state before removal if we reject.
    
    while time.time() < deadline - 1.0:
        # Try Add
        added = False
        # Find any s with count 0
        # We can pick randomly from zero-count sets to avoid bias
        zero_sets = [s for s, c in conflict_count.items() if c == 0 and s not in InF]
        if zero_sets:
            m = rng.choice(zero_sets)
            # Add m
            F.append(m)
            InF.add(m)
            
            # Update counts
            for u in F[:-1]:
                I = u & m
                for s in all_sets:
                    if s in InF: continue
                    if (s & u) == I and (s & m) == I:
                        conflict_count[s] += 1
            added = True
            if len(F) > best_size:
                best_size = len(F)
                best_F = F[:]
                
        if not added:
            # Try Swap: Remove random f, then Greedy Add
            if not F: break
            
            # Save state
            old_F = F[:]
            old_InF = InF.copy()
            old_conflict_count = conflict_count.copy()
            old_size = len(F)
            
            # Pick random f
            f_idx = rng.randint(0, len(F)-1)
            f = F[f_idx]
            
            # Remove f
            F.pop(f_idx)
            InF.remove(f)
            
            # Update counts (decrement)
            for u in F:
                I = u & f
                for s in all_sets:
                    if s in InF: continue # s is in new F (which is old F minus f)
                    # Wait, s might be f? No, f is removed.
                    # s is not in InF.
                    if (s & u) == I and (s & f) == I:
                        conflict_count[s] -= 1
                        
            # Greedy Add phase
            # Try to add as many as possible
            # We need to be careful with the loop termination
            while True:
                zero_sets = [s for s, c in conflict_count.items() if c == 0 and s not in InF]
                if not zero_sets:
                    break
                m = rng.choice(zero_sets)
                F.append(m)
                InF.add(m)
                
                for u in F[:-1]:
                    I = u & m
                    for s in all_sets:
                        if s in InF: continue
                        if (s & u) == I and (s & m) == I:
                            conflict_count[s] += 1
                
            new_size = len(F)
            
            if new_size > old_size:
                # Accept
                if new_size > best_size:
                    best_size = new_size
                    best_F = F[:]
            else:
                # Reject (or accept with prob 0.5 if equal to explore?)
                # Let's reject if smaller, accept if equal with prob 0.5
                if new_size < old_size or (new_size == old_size and rng.random() > 0.5):
                    # Restore
                    F = old_F
                    InF = old_InF
                    conflict_count = old_conflict_count
                else:
                    # Accept equal size (shuffle)
                    if new_size > best_size:
                        best_size = new_size
                        best_F = F[:]
                        
    # Convert best_F to output format
    result = []
    for mask in best_F:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
