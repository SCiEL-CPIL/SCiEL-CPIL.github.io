import time
import random

def construct():
    deadline = time.time() + 55  # Safety margin
    max_val = 256
    max_size = 60

    def validate_and_build(candidates, deadline, used_set=None):
        """
        Greedily add candidates that don't cause cross-cardinality sum collisions.
        candidates: iterable of ints to try
        deadline: time limit
        used_set: set of already used ints (to avoid duplicates, though candidates should be unique)
        """
        if used_set is None:
            used_set = set()
            
        # sums[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
        # We use Python integers as bitsets.
        # To keep bitwise ops fast, we should mask to a reasonable size.
        # Max sum for k elements is k * 256.
        # We will update a mask based on current size.
        
        sums = [0] * (max_size + 1)
        sums[0] = 1  # bit 0 set
        current_max_sum = 0
        result = []
        used = set(used_set)

        for x in candidates:
            if time.time() > deadline:
                break
            if x in used:
                continue
            
            # Calculate new bits contributed by x
            # new_sums[k] = sums[k] | (sums[k-1] << x)
            # We only care about the NEW bits: (sums[k-1] << x) & ~sums[k]
            # And we must ensure these new bits don't collide with ANY sums[j] for j != k.
            
            # To be efficient, let's first check if x is "safe" without updating sums permanently yet.
            # We can compute the potential new bits for each k.
            
            # Optimization: If x is very large, it might not collide with small sums.
            # But we must be exact.
            
            valid = True
            new_sums = sums[:] # Shallow copy, but we will overwrite indices we touch
            # Actually, we only need to check validity. If valid, we update.
            
            # Determine the max k we need to check.
            # Current size is len(result). New size will be len(result)+1.
            # k ranges from 1 to len(result)+1.
            
            # We need to know if (sums[k-1] << x) intersects with any sums[j] (j!=k) 
            # that are NOT already in sums[k] (since sums[k] is already disjoint from sums[j] for j!=k by invariant).
            # Wait, the invariant is that sums[k] and sums[j] are disjoint for all k!=j.
            # So if we add new bits to sums[k], we only need to check if these new bits intersect with sums[j] for j!=k.
            # The new bits are `(sums[k-1] << x) & ~sums[k]`.
            # If these new bits are 0, no change to sums[k], so no new collision introduced by x for this k.
            # If non-zero, check against all sums[j] (j!=k).
            
            # To speed up, we can pre-compute a "union of all other sums" bitset?
            # Let `other_sums` be the OR of all sums[j] for j != k.
            # But `other_sums` depends on k.
            # However, `sums[j]` are disjoint.
            # Let `all_sums` be OR of all sums[j].
            # Then `sums[j]` for j!=k is `all_sums ^ sums[k]`.
            # So we check: `new_bits & (all_sums ^ sums[k]) == 0`.
            
            # Let's maintain `all_sums` as well.
            all_sums = 0
            for s in sums:
                all_sums |= s
                
            # Check validity
            for k in range(1, len(result) + 2):
                if k > max_size: break
                new_bits = (sums[k-1] << x) & ~sums[k]
                if new_bits:
                    # Check collision with other cardinalities
                    # Other sums are all_sums XOR sums[k]
                    # But wait, sums[k] might be 0.
                    # Collision if new_bits shares any bit with sums[j] for j != k.
                    # This is equivalent to new_bits & (all_sums & ~sums[k]) != 0
                    # Since sums are disjoint, all_sums & ~sums[k] is exactly the union of all other sums.
                    if new_bits & (all_sums & ~sums[k]):
                        valid = False
                        break
            
            if valid:
                # Update sums
                for k in range(1, len(result) + 2):
                    if k > max_size: break
                    sums[k] |= (sums[k-1] << x)
                
                # Update all_sums
                all_sums = 0
                for s in sums:
                    all_sums |= s
                    
                result.append(x)
                used.add(x)
                
                if len(result) >= max_size:
                    break

        return result

    def try_extend(best, deadline):
        """Try to add more elements to the best set."""
        if len(best) >= max_size:
            return best
            
        used = set(best)
        remaining = [x for x in range(1, 257) if x not in used]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any remaining element
            # Sort remaining by some heuristic? Maybe smallest first to keep sums low?
            # Or largest?
            # Let's try a few passes with different orders
            orders = [
                sorted(remaining),
                sorted(remaining, reverse=True),
                remaining
            ]
            
            for order in orders:
                if time.time() > deadline:
                    break
                new_best = validate_and_build(order, deadline, used_set=used)
                if len(new_best) > len(best):
                    best = new_best
                    used = set(best)
                    remaining = [x for x in range(1, 257) if x not in used]
                    improved = True
                    if len(best) >= max_size:
                        break
        return best

    def local_search(best, deadline):
        """
        Swap elements to potentially allow adding more elements later.
        Or simply find a valid set of same size that is "better" (e.g. smaller sums) 
        to facilitate extension.
        """
        if len(best) >= max_size:
            return best
            
        used = set(best)
        remaining = [x for x in range(1, 257) if x not in used]
        
        # Heuristic: try to replace larger elements with smaller ones to keep sums compact
        # This might allow more elements to fit.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle remaining to try diverse swaps? Or systematic?
            # Systematic: for each element in best, try replacing with each in remaining
            # This is O(N^2) checks. N=60, Remaining ~200. 12000 checks.
            # Each check is O(N * BitOp). BitOp is fast.
            
            # To make it faster, we can prioritize swaps that reduce the total sum or max sum.
            
            for i, old_val in enumerate(best):
                if time.time() > deadline:
                    break
                for new_val in remaining:
                    if new_val == old_val:
                        continue
                    
                    # Only consider swaps that might help: e.g. new_val < old_val
                    # Or just try all?
                    # Let's try new_val < old_val first to compact the set.
                    if new_val >= old_val:
                        continue
                        
                    test_set = best[:]
                    test_set[i] = new_val
                    test_set.sort() # Keep sorted for consistency? Not strictly needed for DP
                    
                    # Validate test_set
                    # Reusing validate_and_build logic but for a fixed set?
                    # It's faster to just run the DP check on the specific set.
                    
                    valid = True
                    sums = [0] * (max_size + 1)
                    sums[0] = 1
                    all_sums = 1
                    for val in test_set:
                        # Update sums
                        # We need to check collisions during construction? 
                        # No, we can just build the final sums and check if any overlap.
                        # But building sums is O(N^2).
                        # We can do it incrementally.
                        
                        # Actually, let's just use the check from validate_and_build 
                        # but for the whole set at once?
                        # No, the invariant holds step by step.
                        
                        # Efficient check for a fixed set:
                        # Build sums. Then check if any sums[k] & sums[j] != 0 for k!=j.
                        
                        new_sums = sums[:]
                        for k in range(1, len(test_set) + 1):
                            new_sums[k] = sums[k] | (sums[k-1] << val)
                        sums = new_sums
                        
                        # Check for collisions in the final state?
                        # No, collisions must be checked at each step?
                        # Actually, if the final sums are disjoint, were they disjoint at each step?
                        # Yes, because sums only grow. If sum_s in final S_k and sum_s in final S_j (k!=j), 
                        # then they were in the previous steps too?
                        # Not necessarily. A sum might be added to S_k in step 5 and S_j in step 6.
                        # But if they are in the final set, they exist.
                        # The condition is "Whenever two subsets have different cardinalities their sums must differ".
                        # This is a property of the FINAL set.
                        # So we just need to check if the final `sums` list has disjoint entries.
                        
                        # Wait, is it possible that a sum is generated in S_k and S_j only if both elements are present?
                        # Yes. The DP accumulates all subsets.
                        # So checking the final disjointness is sufficient and necessary.
                        
                    # Check disjointness
                    for k in range(len(test_set) + 1):
                        for j in range(k + 1, len(test_set) + 1):
                            if sums[k] & sums[j]:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Accept swap
                        best = test_set
                        used = set(best)
                        remaining = [x for x in range(1, 257) if x not in used]
                        improved = True
                        # Try to extend immediately
                        best = try_extend(best, deadline)
                        used = set(best)
                        remaining = [x for x in range(1, 257) if x not in used]
                        break
                if improved:
                    break
        return best

    # Initial builds
    # 1. Descending (largest first)
    res1 = validate_and_build(range(256, 0, -1), deadline)
    
    # 2. Ascending
    res2 = validate_and_build(range(1, 257), deadline)
    
    # 3. Randomized
    best_rand = []
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline:
            break
        cands = list(range(1, 257))
        random.shuffle(cands)
        res = validate_and_build(cands, deadline)
        if len(res) > len(best_rand):
            best_rand = res

    best = max([res1, res2, best_rand], key=len)
    
    # Extend
    best = try_extend(best, deadline)
    
    # Local Search
    if time.time() < deadline - 5:
        best = local_search(best, deadline)
        
    # Final extend
    best = try_extend(best, deadline)

    if not best:
        return [1]
        
    return [int(x) for x in best]
