import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Prescribed elements
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    # Helper: check if we can add element x to set a (with existing sums)
    def can_add(a, sums, x):
        for y in a:
            s = x + y
            if s in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def get_new_sums(a, x):
        return {x + y for y in a} | {2 * x}
    
    def compute_sums(a):
        s = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                s.add(x + y)
        return s
        
    def compute_conflicts(a, sums):
        """
        Returns a set of values x such that x cannot be added to a.
        x is forbidden if:
        1. x + y == s for some y in a, s in sums
        2. 2*x == s for some s in sums
        """
        forbidden = set()
        for y in a:
            for s in sums:
                x = s - y
                if 1 <= x <= 284:
                    forbidden.add(x)
        for s in sums:
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= 284:
                    forbidden.add(x)
        return forbidden

    def greedy_expand_fast(initial_list, order, forbidden_cache=None):
        a = list(initial_list)
        sums = compute_sums(a)
        a_set = set(a)
        forbidden = compute_conflicts(a, sums)
        
        for x in order:
            if x in a_set:
                continue
            if x in forbidden:
                continue
            
            # Check is effectively done by 'forbidden' set
            # But we must ensure x itself isn't already in a (handled by a_set)
            # And that adding x doesn't create internal conflicts? 
            # Wait, 'forbidden' contains x if x+y=s or 2x=s.
            # If x is not in forbidden, can we always add it?
            # Yes, because the only constraints are sums involving x.
            # Sums involving x are x+y for y in a, and 2x.
            # If none of these are in 'sums', it's safe.
            
            a.append(x)
            a_set.add(x)
            new_sums = get_new_sums(a, x)
            sums |= new_sums
            
            # Update forbidden set incrementally?
            # New forbidden values come from new sums.
            # For each new sum s_new:
            #   x_new = s_new - y for all y in a (including x)
            #   x_new = s_new / 2
            # This might be expensive if we do it fully. 
            # Given small size, recomputing or just checking 'x in forbidden' 
            # with a stale forbidden set is dangerous.
            # Let's just recompute forbidden if we want to be safe, 
            # or better: just check can_add if we don't trust the fast path?
            # Actually, the fast path relies on 'forbidden' being up to date.
            # Let's update forbidden incrementally.
            
            # Add x to forbidden? No, x is now in a.
            # But future candidates y' might conflict with x.
            # The condition for y' being forbidden is y' + z = s for z in a, s in sums.
            # Since x is now in a, any y' such that y' + x = s (for s in sums) is forbidden.
            # Also y' + x = 2*y' (if y'=x, impossible) ... wait.
            # The conflicts are symmetric.
            # If we add x, new sums are S_new.
            # For any future candidate c:
            # c is forbidden if c + z = s for any z in a, s in sums.
            # This includes z=x and s in S_old, and z in a_old and s in S_new.
            
            # Let's just recompute forbidden. It's O(|a| * |sums|). 
            # |a| ~ 20, |sums| ~ 200. 4000 ops. Fast enough.
            forbidden = compute_conflicts(a, sums)
            
        return a

    # Strong local search: try removing 1-4 elements and greedily refilling
    def strong_local_search(a, max_rounds=50):
        a = list(a)
        sums = compute_sums(a)
        a_set = set(a)
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            
            # First, try to add any possible elements using fast greedy
            # We can just use greedy_expand_fast with all remaining candidates
            candidates = [x for x in range(1, 285) if x not in a_set]
            if candidates:
                # Try to add as many as possible
                # We don't have a specific order here, so random or sorted?
                # Sorted ascending is usually good for packing small numbers?
                # Or random?
                # Let's try a few orders if we have time, but for speed, just one.
                # Actually, greedy_expand_fast takes an order.
                # Let's just use the fast add logic directly.
                
                # Re-implement fast add loop for this specific case
                forbidden = compute_conflicts(a, sums)
                changed = True
                while changed:
                    changed = False
                    for cand in candidates:
                        if cand in a_set:
                            continue
                        if cand in forbidden:
                            continue
                        
                        # Add it
                        a.append(cand)
                        a_set.add(cand)
                        new_sums = get_new_sums(a, cand)
                        sums |= new_sums
                        forbidden = compute_conflicts(a, sums)
                        changed = True
                        # Note: candidates list might contain elements now in a_set, 
                        # but we check 'cand in a_set'. 
                        # Also, adding one element might forbid others in candidates.
                        # The 'forbidden' update handles this.
                        # We break inner loop? No, we continue to see if others are still valid.
                        # But if we added cand, it might have made another cand2 invalid.
                        # The 'forbidden' set is updated.
                        # So we can just iterate through all candidates again?
                        # That's O(N^2). N=284. 80k checks. Fast.
                    
            # If we added anything, we improved
            # (We don't track len before/after easily in this inline block, 
            # so let's assume if 'changed' was True, we improved)
            # Actually, let's just check len.
            
            # Wait, the above loop might not be strictly "improved" if it added 0.
            # Let's restart the logic to be cleaner.
            
            # 1. Try to fill up
            prev_len = len(a)
            candidates = [x for x in range(1, 285) if x not in a_set]
            if candidates:
                forbidden = compute_conflicts(a, sums)
                # Simple greedy fill
                for cand in candidates:
                    if cand in a_set:
                        continue
                    if cand in forbidden:
                        continue
                    a.append(cand)
                    a_set.add(cand)
                    new_sums = get_new_sums(a, cand)
                    sums |= new_sums
                    forbidden = compute_conflicts(a, sums)
            
            if len(a) > prev_len:
                improved = True
                continue # Try to fill more in next round? Or just break?
                # Usually filling is monotonic. If we can't add more, we stop.
                # But maybe a different order would allow more? 
                # The 'forbidden' set depends on the set, not the order of addition.
                # So if we can't add any more, we are stuck.
            
            # 2. If not improved (or even if we just filled, maybe we want to try removals?)
            # If we are stuck (len didn't increase in this round's fill phase), try removals.
            # Actually, if len increased, we should try to fill again? 
            # The fill phase above iterates through ALL candidates. 
            # So if it finished, no more can be added in that specific "one pass" logic?
            # Wait, the logic `for cand in candidates` iterates once.
            # If adding cand1 forbids cand2, and cand2 comes later, cand2 is skipped.
            # If cand2 came earlier, it would have been added, and cand1 skipped.
            # So the result depends on order.
            # To be robust, we should try a few orders for the fill phase too.
            # But for now, let's assume the fill phase is "good enough" and move to removals.
            
            if not improved:
                # Try removing 1-4 non-fixed elements and refilling
                non_fixed_indices = [i for i in range(len(a)) if a[i] not in fixed_set]
                
                if not non_fixed_indices:
                    break
                
                # We will try several random removal sets
                # Sizes: 1, 2, 3, 4
                removal_sizes = [1, 2, 3, 4]
                # Weighted towards smaller removals?
                
                best_candidate = None
                best_candidate_len = len(a)
                
                # Number of trials
                num_trials = 20
                
                for trial in range(num_trials):
                    if time.time() > deadline:
                        break
                    
                    # Pick a size
                    size = removal_sizes[trial % len(removal_sizes)]
                    if len(non_fixed_indices) < size:
                        size = len(non_fixed_indices)
                        if size == 0:
                            break
                    
                    # Pick random indices
                    removed_indices = set(random.sample(non_fixed_indices, size))
                    removed_values = {a[i] for i in removed_indices}
                    
                    temp_a = [x for x in a if x not in removed_values]
                    temp_sums = compute_sums(temp_a)
                    temp_set = set(temp_a)
                    
                    # Greedily refill
                    # Use a random order for refill to explore different local minima
                    refill_candidates = [x for x in range(1, 285) if x not in temp_set]
                    random.shuffle(refill_candidates)
                    
                    temp_forbidden = compute_conflicts(temp_a, temp_sums)
                    
                    for cand in refill_candidates:
                        if cand in temp_set:
                            continue
                        if cand in temp_forbidden:
                            continue
                        
                        temp_a.append(cand)
                        temp_set.add(cand)
                        new_sums = get_new_sums(temp_a, cand)
                        temp_sums |= new_sums
                        temp_forbidden = compute_conflicts(temp_a, temp_sums)
                    
                    if len(temp_a) > best_candidate_len:
                        best_candidate_len = len(temp_a)
                        best_candidate = (temp_a, temp_sums, temp_set)
                        
                        # If we hit the theoretical max (20?), we can stop?
                        # Max is 20. If we hit 20, we are done.
                        if best_candidate_len >= 20:
                            break
                
                if best_candidate:
                    a, sums, a_set = best_candidate
                    improved = True
                else:
                    break # No improvement found
            
        return a

    # Try multiple greedy strategies with different orderings
    best = []
    
    # Strategy 1: Ascending order
    a1 = greedy_expand_fast(fixed_elements, range(12, 285))
    if len(a1) > len(best):
        best = a1
    
    # Strategy 2: Descending order
    a2 = greedy_expand_fast(fixed_elements, range(284, 11, -1))
    if len(a2) > len(best):
        best = a2
    
    # Strategy 3: Randomized greedy with many restarts
    random.seed(42)
    n_restarts = 200 # Reduced count to allow more time for local search
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        candidates = list(range(12, 285))
        
        # Different bias strategies
        if restart % 4 == 0:
            random.shuffle(candidates)
        elif restart % 4 == 1:
            # Bias toward larger
            candidates.sort(key=lambda x: random.random() * 50 + x)
        elif restart % 4 == 2:
            # Bias toward smaller
            candidates.sort(key=lambda x: random.random() * 50 - x)
        else:
            # Random
            random.shuffle(candidates)
        
        a3 = greedy_expand_fast(fixed_elements, candidates)
        
        # Apply local search
        a3_improved = strong_local_search(a3, max_rounds=15)
        
        if len(a3_improved) > len(best):
            best = a3_improved
            
            # If we found a very good solution, we can break early?
            # If len(best) == 20, we are optimal.
            if len(best) >= 20:
                break

    # Final polish: apply strong local search to the best
    # Use more rounds for the final best
    best = strong_local_search(best, max_rounds=50)
    
    # One more round of random restarts focusing on the neighborhood of the best
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start from best, remove 1-2 random non-fixed elements, then greedy refill
        a = list(best)
        non_fixed = [x for x in a if x not in fixed_set]
        
        if non_fixed:
            # Remove 1 random element
            idx = random.randint(0, len(non_fixed) - 1)
            removed_val = non_fixed[idx]
            a = [x for x in a if x != removed_val]
        
        # Greedy refill
        sums = compute_sums(a)
        a_set = set(a)
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        # Use fast fill
        forbidden = compute_conflicts(a, sums)
        for cand in candidates:
            if cand in a_set:
                continue
            if cand in forbidden:
                continue
            a.append(cand)
            a_set.add(cand)
            new_sums = get_new_sums(a, cand)
            sums |= new_sums
            forbidden = compute_conflicts(a, sums)
            
        a_improved = strong_local_search(a, max_rounds=20)
        
        if len(a_improved) > len(best):
            best = a_improved
            if len(best) >= 20:
                break

    return best
