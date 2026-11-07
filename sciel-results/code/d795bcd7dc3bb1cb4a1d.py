import time
import random

def construct():
    deadline = time.time() + 52  # safety margin before 60s limit (evaluator kills at 90s, but keep margin)
    max_val = 256
    max_size = 60

    # We need to find a set S of distinct ints in [1,256]
    # such that for any two subsets A, B with |A| != |B|, sum(A) != sum(B)
    # The empty subset has sum 0 and cardinality 0.

    # Strategy: Greedily build the set, maintaining bitsets of achievable sums per cardinality
    # When adding element x, new sums at cardinality k are (old sums at k-1) + x
    # We must check that these don't collide with existing sums at any other cardinality

    def validate_and_build(candidates, deadline):
        """Greedily add candidates that don't cause cross-cardinality sum collisions."""
        # sums[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
        # Max possible sum: at most 60 * 256 = 15360, so we need bits up to ~15360
        max_sum = 60 * 256 + 1
        mask = (1 << max_sum) - 1
        sums = [0] * (max_size + 1)
        sums[0] = 1  # bit 0 set: empty subset has sum 0

        result = []
        for x in candidates:
            if x < 1 or x > 256:
                continue
            # Check if x is already in result
            if x in result:
                continue

            # Try adding x: new sums at cardinality k = (sums[k-1] shifted by x)
            # Check for collisions with existing sums at DIFFERENT cardinalities
            valid = True
            new_sums = [0] * (max_size + 1)
            new_sums[0] = sums[0]
            for k in range(1, len(result) + 2):
                new_sums[k] = sums[k] | ((sums[k-1] << x) & mask)

            # Check: for each cardinality k, new_sums[k] must not overlap with new_sums[j] for j != k
            # But we only need to check the NEW bits added by x
            # New bits at cardinality k: (sums[k-1] << x) & ~(sums[k])
            for k in range(1, len(result) + 2):
                new_bits = (sums[k-1] << x) & ~(sums[k]) & mask
                if new_bits == 0:
                    continue
                # Check if new_bits collide with any other cardinality's sums
                for j in range(max_size + 1):
                    if j == k:
                        continue
                    if new_bits & sums[j]:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                result.append(x)
                sums = new_sums
                if len(result) >= max_size:
                    break
                if time.time() > deadline:
                    break

        return result

    def try_swap_improvement(best, deadline):
        """Try to improve a valid set by swapping elements to free up space or add new ones."""
        if len(best) >= max_size:
            return best
            
        used = set(best)
        remaining = [x for x in range(1, 257) if x not in used]
        
        # Try adding each remaining element
        improved = False
        for x in remaining:
            if time.time() > deadline:
                break
            test_set = best + [x]
            valid = True
            max_sum = 60 * 256 + 1
            mask = (1 << max_sum) - 1
            sums = [0] * (max_size + 1)
            sums[0] = 1
            for val in test_set:
                new_sums = [0] * (max_size + 1)
                new_sums[0] = sums[0]
                for k in range(1, len(test_set) + 1):
                    new_sums[k] = sums[k] | ((sums[k-1] << val) & mask)
                # Check collisions
                for k in range(len(test_set) + 1):
                    for j in range(len(test_set) + 1):
                        if k != j and (new_sums[k] & new_sums[j]):
                            valid = False
                            break
                    if not valid:
                        break
                sums = new_sums
                if not valid:
                    break
            if valid:
                best = test_set
                used.add(x)
                improved = True
                if len(best) >= max_size:
                    break
        
        return best

    # Try multiple candidate orderings
    # Ordering 1: ascending
    result1 = validate_and_build(range(1, 257), deadline)
    
    # Ordering 2: descending
    result2 = validate_and_build(range(256, 0, -1), deadline)
    
    # Ordering 3: Try structured sets (powers of 2 scaled, etc.)
    # Mix: use large gaps
    structured1 = list(range(1, 257, 5))  # every 5th
    result3 = validate_and_build(structured1, deadline)
    
    # Ordering 4: Large values first for more spacing
    large_first = sorted(range(129, 257), reverse=True) + sorted(range(1, 129))
    result4 = validate_and_build(large_first, deadline)
    
    # Ordering 5: Random shuffles (multiple)
    random.seed(42)
    best_random = []
    for _ in range(20):
        if time.time() > deadline:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        res = validate_and_build(candidates, deadline)
        if len(res) > len(best_random):
            best_random = res
            if len(best_random) >= max_size:
                break
    
    # Pick the best
    best = max([result1, result2, result3, result4, best_random], key=len)
    
    # Try to extend with remaining values
    best = try_swap_improvement(best, deadline)
    
    # If we have time, try a local search / simulated annealing step
    # This is a refinement: try swapping one element for another to see if we can add more
    if time.time() < deadline - 2:
        # Simple local search: try replacing one element with another from remaining
        used = set(best)
        remaining = [x for x in range(1, 257) if x not in used]
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i, old_val in enumerate(best):
                if time.time() > deadline - 2:
                    break
                for new_val in remaining:
                    if new_val == old_val:
                        continue
                    # Try swapping
                    test_set = best[:]
                    test_set[i] = new_val
                    valid = True
                    max_sum = 60 * 256 + 1
                    mask = (1 << max_sum) - 1
                    sums = [0] * (max_size + 1)
                    sums[0] = 1
                    for val in test_set:
                        new_sums = [0] * (max_size + 1)
                        new_sums[0] = sums[0]
                        for k in range(1, len(test_set) + 1):
                            new_sums[k] = sums[k] | ((sums[k-1] << val) & mask)
                        # Check collisions
                        for k in range(len(test_set) + 1):
                            for j in range(len(test_set) + 1):
                                if k != j and (new_sums[k] & new_sums[j]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        sums = new_sums
                        if not valid:
                            break
                    
                    if valid:
                        # Check if this swap allows adding more elements
                        # For now, just accept if it's valid (size stays same, but might enable future additions)
                        # Actually, swapping doesn't change size. We need to check if it helps.
                        # A better heuristic: prefer swaps that introduce smaller values (to leave room for more)
                        # or just accept any valid swap and then try to add again.
                        best = test_set
                        used = set(best)
                        remaining = [x for x in range(1, 257) if x not in used]
                        improved = True
                        # Try to add more
                        best = try_swap_improvement(best, deadline)
                        used = set(best)
                        remaining = [x for x in range(1, 257) if x not in used]
                        break
                if improved:
                    break

    # Final validation
    if len(best) == 0:
        return [1]
    
    # Ensure all are Python ints
    best = [int(x) for x in best]
    return best
