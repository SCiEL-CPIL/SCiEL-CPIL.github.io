import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 57  # 3s margin for safety within 60s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128
    ALL_VALS = list(range(MAX_VAL + 1))
    
    # Precompute for each value v in [0,128], the bitset of sums in [128,256] it can form
    # with each other value. Actually, we need pair sums.
    # Let's represent the set of achievable sums in [128,256] as a 129-bit integer.
    # For a set A, the sumset S(A) = {a+b : a,b in A, a<=b} intersected with [128,256].
    # We want every element in [128,256] to be in S(A).
    # And we want to minimize max_{s in [128,256]} count_{a,b in A, a<=b, a+b=s}.
    
    # Bitset representation: bit i (0-indexed) corresponds to sum N_LOW + i.
    # For a single element a, the "potential" sums with elements b are a+b.
    # But it's easier to work with the actual set A and compute reps.
    
    def calc_reps_bitset(A):
        """
        Returns (reps_array, covered_mask) where reps_array[s-N_LOW] is the count,
        and covered_mask is a boolean array of length RANGE_SIZE.
        """
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_list = list(A)
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            for j in range(i, n):
                aj = A_list[j]
                s = ai + aj
                if N_LOW <= s <= N_HIGH:
                    reps[s - N_LOW] += 1
        covered = reps > 0
        return reps, covered
    
    def is_covered(covered):
        return np.all(covered)
    
    def get_peak(reps):
        return int(np.max(reps))
    
    # Phase 1: Greedy construction from scratch
    # Start with empty set, or a small seed.
    # To ensure coverage, we can use a set-cover-like greedy.
    # Each element a in [0,128] contributes to sums with existing elements.
    # This is complex because the contribution of a depends on the current set.
    
    # Alternative Phase 1: Start with full set [0,128] and prune.
    # This is the "greedy peeling" strategy mentioned in findings, which improved score from 3.3 to 5.5.
    # Let's try that.
    
    A = set(ALL_VALS)
    reps, covered = calc_reps_bitset(A)
    peak = get_peak(reps)
    
    # Pruning: Remove elements that don't break coverage, preferably those that reduce peak or don't increase it.
    # Since we start with full set, peak is likely very high.
    # Removing an element reduces the number of pairs, so peak should decrease or stay same.
    # We want to remove as many as possible while keeping coverage.
    
    # Order of removal: Try removing elements that are "redundant" for coverage.
    # An element a is redundant if removing it doesn't make any sum in [128,256] have count 0.
    # To be efficient, we can iterate.
    
    # To make pruning more effective, we can try to remove elements in a specific order.
    # Elements in the middle (around 64) participate in more sums in [128,256] than extremes?
    # Actually, 0+128=128, 128+128=256.
    # 64+64=128, 64+128=192.
    # Let's just shuffle and try removals.
    
    # More aggressive: Iteratively find an element that can be removed without breaking coverage.
    # If multiple, pick one that minimizes the new peak? Or just any.
    # Since we want to minimize peak, and removal generally lowers peak, we should remove as many as possible.
    
    # However, removing too many might make it hard to reduce peak further via swaps.
    # So we should prune to a "minimal" covering set, then optimize peak via swaps.
    
    # Let's prune until no single removal preserves coverage.
    # This gives a minimal set (in terms of inclusion).
    
    # But wait, minimal set might have high peak. We might want a slightly larger set with lower peak.
    # So after pruning to minimal, we can try adding elements back if it reduces peak.
    
    # Let's start with pruning.
    
    # For speed, we can use a bitset for coverage.
    # Let's represent the "coverage requirement" as a set of sums that must be covered.
    # When we remove a, we check if any sum s in [128,256] loses its last representation involving a.
    # This is equivalent to: for all s, if s was covered only by pairs involving a, then s becomes uncovered.
    
    # Efficient check:
    # For current A, compute reps.
    # For candidate removal a, compute how many pairs involving a contribute to each s.
    # If for some s, reps[s] == count_pairs_involving_a(s), then removing a breaks coverage for s.
    
    # Let's implement this.
    
    def can_remove(A, a, reps):
        """
        Check if removing a from A preserves coverage.
        reps is the current representation count array.
        """
        # Count pairs involving a
        # Pairs are (a, b) where b in A, b >= a (if we consider unordered with a<=b)
        # Actually, pairs are (x, y) with x <= y.
        # If we remove a, we lose pairs (a, b) for all b in A with a <= b, and (b, a) for b < a?
        # No, the pair is unordered. The set of pairs is {{x,y} : x,y in A, x<=y}.
        # Removing a removes all pairs {a, b} for b in A.
        # Note: if b=a, the pair is {a,a}.
        
        # Let's count for each s, how many pairs {a, b} with b in A sum to s.
        # This is: for each b in A, if a+b in [128,256], then s=a+b.
        # But we must be careful: the pair {a,b} is counted once.
        # In the original reps, the pair {a,b} contributes 1 to reps[a+b-N_LOW].
        
        # So, for each b in A:
        #   s = a + b
        #   if N_LOW <= s <= N_HIGH:
        #       if reps[s-N_LOW] == 1: # Only this pair covers s?
        #           return False
        #       # else, reps[s-N_LOW] > 1, so removing this pair still leaves s covered.
        # But wait, there might be other pairs involving a? No, we're checking if ANY sum becomes uncovered.
        # A sum s becomes uncovered if ALL its representing pairs involve a.
        # This is harder to check directly.
        
        # Alternative: Compute new reps.
        # Since |A| is at most 129, and we're doing this many times, it might be slow.
        # But we can optimize:
        # The change in reps when removing a is:
        # For each b in A:
        #   s = a + b
        #   if N_LOW <= s <= N_HIGH:
        #       new_reps[s-N_LOW] = reps[s-N_LOW] - 1
        #       if new_reps[s-N_LOW] < 0: error
        #       if new_reps[s-N_LOW] == 0:
        #           return False # Coverage broken
        # return True
        
        # This is O(|A|) per check.
        # Total pruning: O(|A|^2) checks, each O(|A|). So O(|A|^3).
        # |A| starts at 129. 129^3 ~ 2 million. Should be fast enough.
        
        for b in A:
            s = a + b
            if N_LOW <= s <= N_HIGH:
                if reps[s - N_LOW] == 1:
                    return False
        return True
    
    # Pruning loop
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Shuffle to avoid bias
        A_list = list(A)
        random.shuffle(A_list)
        
        for a in A_list:
            if time.time() > deadline:
                break
            if a not in A:
                continue
            if can_remove(A, a, reps):
                # Remove a
                A.remove(a)
                # Update reps
                for b in A:
                    s = a + b
                    if N_LOW <= s <= N_HIGH:
                        reps[s - N_LOW] -= 1
                # Also, the pair (a,a) was counted once.
                # In the loop above, when b=a, s=2a.
                # If a is still in A? No, we removed a.
                # Wait, the loop `for b in A` is after `A.remove(a)`.
                # So b iterates over the NEW A.
                # The pairs removed are {a, b} for b in NEW A.
                # What about {a, a}?
                # {a, a} sums to 2a.
                # In the original reps, {a,a} contributed to 2a.
                # In the new reps, {a,a} is gone.
                # Does the loop `for b in A` (new A) include b=a? No, a is removed.
                # So we missed subtracting for {a,a}.
                
                # Let's fix the update logic.
                # Pairs removed: {a, b} for all b in OLD A.
                # Since a is in OLD A, b can be a.
                # So we should iterate over OLD A.
                
                # Let's redo the update.
                # We already did A.remove(a). So A is NEW A.
                # We need to subtract for b in OLD A.
                # OLD A = NEW A U {a}.
                # So we subtract for b in NEW A, and also for b=a.
                
                # The loop above did: for b in A (NEW A): reps[a+b] -= 1
                # We need to also do: s=2a; if N_LOW<=s<=N_HIGH: reps[s-N_LOW] -= 1
                
                s_aa = 2 * a
                if N_LOW <= s_aa <= N_HIGH:
                    reps[s_aa - N_LOW] -= 1
                    
                changed = True
                # Break and restart? Or continue?
                # If we continue, the reps array is updated, so subsequent checks are valid.
                # But the order of removal might matter.
                # Let's continue with the updated reps.
                
    # After pruning, we have a minimal covering set (no single element can be removed).
    # Now, try to improve peak via swaps.
    
    # Also, try adding elements back if it reduces peak.
    # This is a "swap" or "add" step.
    
    # Let's do a local search:
    # 1. Try swaps: remove a, add b.
    # 2. Try additions: add b (if it reduces peak).
    # 3. Try removals: (already done, but maybe after swaps we can remove more).
    
    best_A = set(A)
    best_reps = reps.copy()
    best_peak = get_peak(best_reps)
    
    # Local search
    # To speed up, we can limit the number of iterations or time.
    
    # We'll use a simple hill climbing with restarts if stuck.
    
    # For each iteration:
    #   Try all possible swaps (a in A, b not in A).
    #   If a swap reduces peak, accept it.
    #   If no swap reduces peak, try additions.
    #   If no addition reduces peak, break.
    
    # To avoid O(|A| * (129-|A|)) which is ~ 129*129/2 ~ 8000 swaps per iteration,
    # and each swap requires O(|A|) to update reps, total ~ 1 million ops per iteration.
    # With 60s limit, we can do many iterations.
    
    # But we need to be careful with time.
    
    # Let's implement a faster swap check.
    # When swapping a -> b:
    #   Remove a: subtract contributions of {a, x} for x in A.
    #   Add b: add contributions of {b, x} for x in A \ {a}.
    #   Net change:
    #     For each x in A \ {a}:
    #       s1 = a + x -> reps[s1] -= 1
    #       s2 = b + x -> reps[s2] += 1
    #     Also, pair {a,a} was removed, pair {b,b} is added.
    #       s_aa = 2a -> reps[s_aa] -= 1
    #       s_bb = 2b -> reps[s_bb] += 1
    #     Note: if x=a in the loop, we handle {a,a} separately.
    #     So loop over x in A \ {a}.
    
    # We can precompute the "delta" for each swap.
    
    # Let's try a few iterations.
    
    for _ in range(100):  # Max 100 iterations
        if time.time() > deadline - 2:  # Leave 2s for final check
            break
        
        improved = False
        A_list = list(best_A)
        non_A = [v for v in ALL_VALS if v not in best_A]
        
        # Shuffle for diversity
        random.shuffle(A_list)
        random.shuffle(non_A)
        
        for a in A_list:
            if time.time() > deadline - 2:
                break
            for b in non_A:
                if time.time() > deadline - 2:
                    break
                
                # Simulate swap a -> b
                # Check if it reduces peak.
                # We need to compute the new peak.
                # Instead of full recomputation, we can track changes.
                
                # But it's complex. Let's just recompute if |A| is small.
                # Or, we can do a quick check:
                # The peak can only increase if the new element b creates many high-count sums.
                # Let's just do a full recompute for simplicity, since |A| is likely small after pruning.
                
                new_A = best_A - {a}
                new_A.add(b)
                
                # Quick check: is new_A covered?
                # We can check coverage by seeing if any sum becomes 0.
                # But we need the actual reps to get the peak.
                
                new_reps, new_covered = calc_reps_bitset(new_A)
                
                if not is_covered(new_covered):
                    continue
                
                new_peak = get_peak(new_reps)
                
                if new_peak < best_peak:
                    best_A = new_A
                    best_reps = new_reps
                    best_peak = new_peak
                    improved = True
                    # Restart inner loops
                    A_list = list(best_A)
                    non_A = [v for v in ALL_VALS if v not in best_A]
                    break
            
            if improved:
                break
        
        if not improved:
            # Try additions
            added = False
            for b in non_A:
                if time.time() > deadline - 2:
                    break
                new_A = best_A | {b}
                new_reps, new_covered = calc_reps_bitset(new_A)
                if is_covered(new_covered):
                    new_peak = get_peak(new_reps)
                    if new_peak < best_peak:
                        best_A = new_A
                        best_reps = new_reps
                        best_peak = new_peak
                        added = True
                        break
            if not added:
                # Try removals again
                removed = False
                for a in list(best_A):
                    if time.time() > deadline - 2:
                        break
                    if can_remove(best_A, a, best_reps):
                        new_A = best_A - {a}
                        # Update reps
                        for b in new_A:
                            s = a + b
                            if N_LOW <= s <= N_HIGH:
                                best_reps[s - N_LOW] -= 1
                        s_aa = 2 * a
                        if N_LOW <= s_aa <= N_HIGH:
                            best_reps[s_aa - N_LOW] -= 1
                        best_A = new_A
                        best_peak = get_peak(best_reps)
                        removed = True
                        break
                if not removed:
                    break
    
    # Final check: ensure coverage
    final_reps, final_covered = calc_reps_bitset(best_A)
    if not is_covered(final_covered):
        # Fallback: return full set if something went wrong
        return ALL_VALS
    
    return list(best_A)

if __name__ == "__main__":
    t0 = time.time()
    A = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, |A|={len(A)}")
    reps = [0]*129
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if 128 <= s <= 256:
                reps[s-128] += 1
    print(f"Covered: {all(r>0 for r in reps)}, Peak: {max(reps)}")
