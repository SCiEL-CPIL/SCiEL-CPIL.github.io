"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50.0  # safety margin before 60s limit
    
    # Strategy: Use a threshold-based approach with optimization
    # {c, c+1, ..., 100} is valid if:
    #   - 2-element max < 200: (100 + 99) = 199 < 200 (always true for c >= 2)
    #   - 3-element min > 200: c + (c+1) + (c+2) > 200 => 3c+3 > 200 => c >= 66
    # So {66,...,100} gives 35 elements, valid.
    
    # Try to do better by adding some smaller elements carefully
    base = set(range(66, 101))  # 35 elements, known valid
    
    # Verify base is valid
    def has_subset_sum_200(s):
        """Check if any subset of s sums to 200 using bitset DP."""
        reachable = 1  # bit 0 set
        for x in s:
            if x <= 200:
                reachable |= (reachable << x)
        return (reachable >> 200) & 1
    
    if has_subset_sum_200(base):
        # Fallback to greedy
        selected, reachable_set = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable_set if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable_set |= new
        return selected
    
    best = set(base)
    best_size = len(best)
    
    # Try adding smaller elements one by one, greedily
    # Process from larger to smaller for better pruning
    candidates = list(range(65, 0, -1))
    
    for x in candidates:
        if time.time() > deadline:
            break
        trial = best | {x}
        if not has_subset_sum_200(trial):
            best = trial
            best_size += 1
    
    # Try to add multiple small elements together
    # Use a more systematic approach: try adding pairs/triples
    remaining = [x for x in range(65, 0, -1) if x not in best]
    
    # Try adding each remaining element
    for x in remaining:
        if time.time() > deadline:
            break
        trial = best | {x}
        if not has_subset_sum_200(trial):
            best = trial
            best_size += 1
    
    # Second pass: try again after additions
    remaining = [x for x in range(65, 0, -1) if x not in best]
    for x in remaining:
        if time.time() > deadline:
            break
        trial = best | {x}
        if not has_subset_sum_200(trial):
            best = trial
            best_size += 1
    
    # Convert to list
    result = list(best)
    if not result:
        result = list(base)
    
    return result
# EVOLVE-BLOCK-END
