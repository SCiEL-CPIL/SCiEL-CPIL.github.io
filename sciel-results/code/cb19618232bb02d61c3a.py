import time
from itertools import combinations

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    max_val = 256
    max_size = 60
    
    # We need to find a set S of distinct ints in [1,256]
    # such that for any two subsets A, B with |A| != |B|, sum(A) != sum(B)
    # The empty subset has sum 0 and cardinality 0.
    
    # Strategy: Greedily build the set, maintaining bitsets of achievable sums per cardinality
    # When adding element x, new sums at cardinality k are (old sums at k-1) + x
    # We must check that these don't collide with existing sums at any other cardinality
    
    best_set = [1]  # Start with smallest
    
    # Try different strategies
    # Strategy 1: Greedy from small to large
    # Strategy 2: Greedy from large to small  
    # Strategy 3: Try structured sets (powers of 2 scaled, etc.)
    
    def validate_and_build(candidates):
        """Greedily add candidates that don't cause cross-cardinality sum collisions."""
        # sums[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
        # Max possible sum: at most 60 * 256 = 15360, so we need bits up to ~15360
        max_sum = 60 * 256 + 1
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
                new_sums[k] = sums[k] | ((sums[k-1] << x) & ((1 << max_sum) - 1))
            
            # Check: for each cardinality k, new_sums[k] must not overlap with new_sums[j] for j != k
            # But we only need to check the NEW bits added by x
            # New bits at cardinality k: (sums[k-1] << x) & ~(sums[k])
            for k in range(1, len(result) + 2):
                new_bits = (sums[k-1] << x) & ~(sums[k]) & ((1 << max_sum) - 1)
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
    
    # Try multiple candidate orderings
    # Ordering 1: ascending
    result1 = validate_and_build(range(1, 257))
    
    # Ordering 2: descending
    result2 = validate_and_build(range(256, 0, -1))
    
    # Ordering 3: Try powers of 2 based
    powers = [2**i for i in range(1, 9) if 2**i <= 256]  # 2,4,8,...,128
    # Mix: use large gaps
    structured1 = list(range(1, 257, 5))  # every 5th
    result3 = validate_and_build(structured1)
    
    # Ordering 4: Large values first for more spacing
    large_first = sorted(range(129, 257), reverse=True) + sorted(range(1, 129))
    result4 = validate_and_build(large_first)
    
    # Pick the best
    best = max([result1, result2, result3, result4], key=len)
    
    # Try to extend with remaining values
    used = set(best)
    remaining = [x for x in range(1, 257) if x not in used]
    for x in remaining:
        if time.time() > deadline:
            break
        test_set = best + [x]
        # Quick validation
        valid = True
        # Rebuild sums for test_set
        max_sum = 60 * 256 + 1
        sums = [0] * (max_size + 1)
        sums[0] = 1
        for val in test_set:
            new_sums = [0] * (max_size + 1)
            new_sums[0] = sums[0]
            for k in range(1, len(test_set) + 1):
                new_sums[k] = sums[k] | ((sums[k-1] << val) & ((1 << max_sum) - 1))
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
            if len(best) >= max_size:
                break
    
    # Final validation
    if len(best) == 0:
        return [1]
    
    # Ensure all are Python ints
    best = [int(x) for x in best]
    return best
