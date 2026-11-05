import time
import random
import itertools
from functools import lru_cache

def construct():
    """
    Constructs a linear triple system on 16 points with a small independence number.
    Strategy: Iterated Local Search (ILS) optimizing the independence number directly.
    1. Initialize with a random greedy packing.
    2. Calculate exact independence number.
    3. Perform local moves (Add, Remove, Swap) to reduce the independence number.
    4. Restart with perturbations if stuck.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    n_points = 16
    points = list(range(n_points))
    
    # Pre-compute all possible triples
    all_triples = list(itertools.combinations(points, 3))
    # Map pair to triple index? Or just use sets.
    # For fast lookup of "is pair used", we maintain a set of used pairs.
    
    def get_used_pairs(triples):
        used = set()
        for a, b, c in triples:
            used.add((a, b))
            used.add((b, a))
            used.add((b, c))
            used.add((c, b))
            used.add((a, c))
            used.add((c, a))
        return used

    def is_valid_addition(triple, used_pairs):
        a, b, c = triple
        if (a, b) in used_pairs: return False
        if (b, c) in used_pairs: return False
        if (a, c) in used_pairs: return False
        return True

    def calculate_independence_number(triples):
        """
        Calculate the exact independence number of the 3-uniform hypergraph.
        Since N=16, we can use a branch and bound or simply check subsets if optimized.
        However, checking all 2^16 subsets is 65536, which is fast.
        For each subset, check if it contains any triple.
        We want the largest subset that contains NO triple.
        """
        # Bitmask representation for speed
        # triple_mask: integer with bits set for points in the triple
        triple_masks = []
        for a, b, c in triples:
            mask = (1 << a) | (1 << b) | (1 << c)
            triple_masks.append(mask)
        
        # If no triples, alpha is 16
        if not triple_masks:
            return n_points
            
        # We iterate through subsets by size, from largest to smallest?
        # Or just check all subsets.
        # To speed up, we can use a recursive search or just brute force all 2^16 masks.
        # 65536 masks. For each mask, check if it contains any triple mask.
        # A subset S contains a triple T if (S_mask & T_mask) == T_mask.
        
        # Optimization: Sort triples by frequency? No, just check.
        # We can precompute for each point mask which triples it contains?
        
        max_alpha = 0
        # Iterate through all possible subsets
        # We can prune: if a subset is independent, all its subsets are independent.
        # We want the MAXIMUM size independent set.
        # Let's use a DP or just simple iteration with pruning.
        
        # Simple iteration is O(2^N * M) where M is num triples.
        # 65536 * ~50 = 3M ops. Very fast in Python if optimized?
        # Let's try to be smart.
        
        # Actually, we can find the maximum independent set using a simple backtracking or just checking all subsets.
        # Given N=16, checking all subsets is feasible if we do it efficiently.
        
        # Let's store triple masks in a set for O(1) lookup? No, we need "is T subset of S".
        # We can precompute for each triple T, the list of supersets? Too many.
        
        # Alternative: 
        # For each subset S (bitmask), check if it is independent.
        # S is independent if no T in triples is a subset of S.
        
        # Let's iterate S from 1 to 2^16 - 1.
        # To find the max size, we can iterate by popcount descending.
        
        # Pre-compute popcount for 16-bit integers?
        # Or just use bin(x).count('1').
        
        # To speed up, we can group triples by their minimum element?
        
        # Let's just do the brute force check for all subsets, but break early if we find one of size k.
        # We want the max k.
        
        # A faster way: 
        # Let V be the set of vertices.
        # We can use a recursive function to find the maximum independent set.
        
        def max_independent_set(remaining_points_mask, available_triples_indices):
            # This might be complex to implement correctly and fast in Python with recursion limits.
            pass
            
        # Let's stick to the subset enumeration but optimized.
        # We want to know if a set S contains a triple.
        # We can precompute a boolean array `is_independent` of size 2^16.
        # is_independent[0] = True
        # For mask from 1 to 2^16 - 1:
        #   is_independent[mask] = True
        #   for each triple T:
        #      if (mask & T_mask) == T_mask:
        #          is_independent[mask] = False
        #          break
        #   if is_independent[mask]:
        #       max_alpha = max(max_alpha, popcount(mask))
        
        # This is O(2^16 * M). 65536 * 50 = 3.2M iterations.
        # In Python, this might take ~1-2 seconds.
        # We will do this calculation multiple times (for each local search step).
        # This might be too slow for 120s if we do hundreds of steps.
        
        # Optimization: Only recalculate alpha when the set of triples changes significantly?
        # Or use a heuristic estimate for local search, and exact calc for final/best tracking.
        
        # Let's implement a faster heuristic for the local search loop:
        # Estimate alpha by finding a large independent set using a greedy algorithm.
        
        def greedy_independent_set(triples, used_pairs):
            """
            Greedily find a large independent set.
            Start with all points.
            Remove points involved in triples? No.
            A set I is independent if no triple is fully contained in I.
            
            Greedy approach:
            1. Start with I = {0..15}.
            2. While there is a triple T fully contained in I:
               Remove one point from T from I. (Which one? Random or lowest degree in I?)
            3. Return I.
            """
            I_mask = (1 << n_points) - 1
            # We need to know which triples are fully contained in I.
            # A triple (a,b,c) is in I if (I_mask & (1<<a)) and (I_mask & (1<<b)) and (I_mask & (1<<c)).
            
            # To be fast, we can just iterate through triples.
            # If a triple is in I, we must remove at least one of its points.
            # To maximize I, we should remove a point that is in many other triples inside I.
            
            # Simple version:
            # Repeat until no triple is in I:
            #   Find a triple T in I.
            #   Pick a point p in T.
            #   Remove p from I.
            
            # This is not guaranteed to find the MAX independent set, but gives a lower bound.
            # We want a LOWER bound on alpha to see if we can improve?
            # Actually, if we find a set of size 7, alpha >= 7.
            # We want to MINIMIZE alpha. So we want to MINIMIZE the size of the largest independent set.
            # If the greedy finds a set of size 7, the true alpha is at least 7.
            # If we want to prove alpha <= 6, we need to show NO set of size 7 is independent.
            # This is hard.
            
            # Let's use the exact calculation but cache it?
            # The set of triples changes by 1 each step.
            # Maybe we can update the independence number incrementally?
            # Too complex.
            
            # Let's assume the exact calculation is ~1s.
            # 120s / 1s = 120 steps.
            # This is very few for local search.
            
            # We need a faster proxy.
            # Proxy: Number of triples. More triples -> lower alpha.
            # But not strictly monotonic.
            
            # Let's try a different heuristic: 
            # Count the number of "uncovered" 6-sets.
            # If all 6-sets are covered (contain a triple), then alpha <= 5.
            # If we minimize the number of uncovered 6-sets, we likely minimize alpha.
            
            return 0 # Placeholder

        # Let's go back to the exact calculation but optimize it.
        # We can use numpy for vectorized operations?
        # Create a numpy array of triple masks.
        # For a given subset mask S, we want to check if any T is subset of S.
        # This is equivalent to: does there exist T such that (S & T) == T?
        
        # Let's use a simple optimization:
        # Pre-compute for each point p, the list of triples containing p.
        # This doesn't help much with subset checking.
        
        # Let's just implement the exact check efficiently.
        
        triple_masks = [ (1<<a) | (1<<b) | (1<<c) for a,b,c in triples ]
        
        # If we have many triples, it's likely alpha is small.
        # We can stop checking subsets of size > current_best_alpha.
        
        best_alpha = n_points
        # Iterate subsets by size descending
        # We can generate all masks and sort by popcount?
        # Or just check all masks.
        
        # To speed up, we can check if a mask is independent by:
        # independent = True
        # for t in triple_masks:
        #    if (mask & t) == t:
        #       independent = False
        #       break
        
        # This is O(M) per mask.
        
        # Let's try to run this. If it's too slow, we'll reduce the number of local search steps.
        
        import numpy as np
        # Convert triple masks to numpy array
        t_arr = np.array(triple_masks, dtype=np.uint32)
        
        # We can't easily vectorize the "subset" check for a single mask against all triples in numpy without broadcasting which is 65536 x M.
        # 65536 * 50 = 3M ops. Numpy is fast.
        # But doing this for EVERY local search step (120 steps) is 360M ops. Might be okay?
        
        # Let's try.
        
        def calc_alpha_exact(triples_list):
            if not triples_list:
                return n_points
            t_masks = [ (1<<a) | (1<<b) | (1<<c) for a,b,c in triples_list ]
            t_arr = np.array(t_masks, dtype=np.uint32)
            
            # Generate all masks 0 to 2^16 - 1
            # We only care about masks with popcount > current best?
            # We don't know current best until we check.
            
            # Let's just check all.
            # Create an array of all masks
            # This is 65536 elements.
            masks = np.arange(1, 1 << n_points, dtype=np.uint32)
            
            # We want to find the max popcount of a mask that is INDEPENDENT.
            # A mask m is independent if NO t in t_arr is a subset of m.
            # (m & t) == t
            
            # This is a bit tricky to vectorize fully without a loop over t_arr or masks.
            # Let's loop over masks, but use numpy for the check against t_arr?
            # For each m in masks:
            #    check = (masks[i] & t_arr) == t_arr
            #    if not check.any():
            #        # m is independent
            #        popcount = bin(m).count('1')
            #        if popcount > best_alpha: best_alpha = popcount
            
            # This is a Python loop of 65536 iterations.
            # Inside, a numpy operation of size M (50).
            # 65536 * 50 ops in numpy is fast (~ms).
            # But the Python loop overhead is 65536 * ~1us = 65ms.
            # Total ~100ms per calculation.
            # 120 steps * 100ms = 12s. This is feasible!
            
            best_alpha = 0
            # We can prune: if we find a mask of size k, we don't need to check masks of size < k?
            # No, we want the MAX. So we can stop if we find one of size 16 (impossible if triples exist).
            # We can iterate masks in order of decreasing popcount to find the max faster?
            # Generating masks in decreasing popcount order is complex.
            
            # Let's just iterate all.
            # Optimization: If best_alpha is already small (e.g., 5), we only need to check masks with popcount > 5.
            
            # Pre-compute popcounts?
            # bin(m).count('1') is fast enough.
            
            for m in masks:
                # Quick check: if popcount(m) <= best_alpha, skip
                # This saves time if best_alpha is high.
                # But initially best_alpha is 0.
                
                # Check independence
                # (m & t_arr) == t_arr
                # If any is True, m is NOT independent.
                is_independent = not np.any((m & t_arr) == t_arr)
                
                if is_independent:
                    pc = bin(m).count('1')
                    if pc > best_alpha:
                        best_alpha = pc
                        # If we hit 16, we can break? No, 16 is max.
            
            return best_alpha

    # --- Main Search Loop ---
    
    best_alpha = n_points
    best_triples = []
    
    # Initial Greedy Packing to get a good start
    def initial_greedy():
        used_pairs = set()
        triples = []
        random.shuffle(all_triples)
        for trip in all_triples:
            if is_valid_addition(trip, used_pairs):
                triples.append(list(trip))
                a, b, c = trip
                used_pairs.add((a, b)); used_pairs.add((b, a))
                used_pairs.add((b, c)); used_pairs.add((c, b))
                used_pairs.add((a, c)); used_pairs.add((c, a))
        return triples

    current_triples = initial_greedy()
    current_alpha = calc_alpha_exact(current_triples)
    
    if current_alpha < best_alpha:
        best_alpha = current_alpha
        best_triples = [t[:] for t in current_triples]
        
    # Local Search
    # Move types:
    # 1. Add a valid triple.
    # 2. Remove a triple.
    # 3. Swap (Remove one, Add one).
    
    # To guide the search, we want to reduce alpha.
    # Adding a triple usually reduces alpha (or keeps it same).
    # Removing a triple usually increases alpha.
    # So we should prioritize ADD moves.
    # If we can't add, we might need to SWAP.
    
    # We will perform a random walk with acceptance criteria:
    # Accept if new_alpha < current_alpha (Improvement)
    # Accept if new_alpha == current_alpha with some probability (Plateau)
    # Reject if new_alpha > current_alpha (Worsening) - unless we want to escape local optima.
    # Since we want to MINIMIZE, we should generally avoid worsening.
    # But sometimes we need to remove a "bad" triple to allow adding two "good" ones.
    # This requires a 2-step move (Remove, then Add).
    # Or a Swap.
    
    # Let's implement a simple Simulated Annealing.
    # Temperature T starts high, decays.
    # Accept with probability exp(-(new_alpha - current_alpha) / T).
    # Since alpha is integer, the delta is integer.
    
    T = 1.0
    cooling = 0.99
    
    steps = 0
    max_steps = 5000 # Limit to ensure time
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Generate a random move
        move_type = random.random()
        
        new_triples = [t[:] for t in current_triples]
        used_pairs = get_used_pairs(new_triples)
        
        if move_type < 0.5:
            # Try to ADD a triple
            # Pick a random triple from all_triples
            # Check if valid
            # If valid, add it
            candidate = random.choice(all_triples)
            if is_valid_addition(candidate, used_pairs):
                new_triples.append(list(candidate))
                # Calculate new alpha
                new_alpha = calc_alpha_exact(new_triples)
                
                # Acceptance
                if new_alpha < current_alpha:
                    current_triples = new_triples
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = [t[:] for t in current_triples]
                elif new_alpha == current_alpha:
                    # Accept plateau moves to explore
                    if random.random() < 0.5:
                        current_triples = new_triples
                        # current_alpha stays same
                # Else reject
            # If not valid, do nothing (or try another?)
            # For efficiency, just skip if invalid.
            
        elif move_type < 0.8:
            # Try to REMOVE a triple
            if new_triples:
                idx = random.randint(0, len(new_triples) - 1)
                new_triples.pop(idx)
                new_alpha = calc_alpha_exact(new_triples)
                
                # Removing a triple likely increases alpha.
                # Only accept if it leads to a better state later?
                # In SA, we might accept worse states.
                delta = new_alpha - current_alpha
                if delta < 0:
                    current_triples = new_triples
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = [t[:] for t in current_triples]
                else:
                    # Accept with prob exp(-delta/T)
                    if delta > 0:
                        prob = np.exp(-delta / T)
                        if random.random() < prob:
                            current_triples = new_triples
                            current_alpha = new_alpha
        else:
            # Try to SWAP
            # Remove one, Add one
            if new_triples:
                idx = random.randint(0, len(new_triples) - 1)
                removed_trip = new_triples[idx]
                new_triples.pop(idx)
                
                # Update used_pairs for the removed triple?
                # We need to check if the new triple is valid against the REMAINING triples.
                # So we must recalculate used_pairs or do it carefully.
                # For simplicity, recalculate used_pairs for the new_triples (after pop).
                used_pairs = get_used_pairs(new_triples)
                
                candidate = random.choice(all_triples)
                if is_valid_addition(candidate, used_pairs):
                    new_triples.append(list(candidate))
                    new_alpha = calc_alpha_exact(new_triples)
                    
                    delta = new_alpha - current_alpha
                    if delta < 0:
                        current_triples = new_triples
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_triples = [t[:] for t in current_triples]
                    elif delta == 0:
                        if random.random() < 0.5:
                            current_triples = new_triples
                    else:
                        prob = np.exp(-delta / T)
                        if random.random() < prob:
                            current_triples = new_triples
                            current_alpha = new_alpha
        
        # Cool down
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor
            
    return best_triples

construct()
