import time
from fractions import Fraction
import random
import math
from collections import defaultdict

def _precompute_conflicts():
    """
    For each x in 1..24, find all subsets Y of [1,24]\{x} such that sum(1/y for y in Y) == 1/x.
    Returns a list of lists of bitmasks.
    Uses Meet-in-the-Middle for speed.
    """
    universe = list(range(1, 25))
    L = 5354228880
    
    conflicts = [[] for _ in range(25)] # 1-indexed
    
    for x in range(1, 25):
        target = L // x
        candidates = []
        for y in universe:
            if y != x:
                candidates.append((y, L // y))
        
        n = len(candidates)
        mid = n // 2
        left = candidates[:mid]
        right = candidates[mid:]
        
        # Generate all subset sums for left part
        # Map: sum -> list of masks
        left_sums = defaultdict(list)
        # Use iterative generation to avoid recursion depth issues and overhead
        # There are 2^mid subsets.
        # mid is at most 12. 2^12 = 4096.
        
        # Initialize with empty subset
        left_sums[0].append(0)
        
        # Iterate through all subsets of left?
        # Or build iteratively.
        # Let's use a list of (sum, mask) and expand.
        # Actually, simple recursion or bit iteration is fine for size 12.
        
        # Let's use bit iteration for clarity and speed in Python?
        # No, simple recursion is fine for 12 items.
        
        def gen_subsets(items, start_idx, current_sum, current_mask, dest_dict):
            if start_idx == len(items):
                dest_dict[current_sum].append(current_mask)
                return
            # Skip
            gen_subsets(items, start_idx + 1, current_sum, current_mask, dest_dict)
            # Take
            y, val = items[start_idx]
            gen_subsets(items, start_idx + 1, current_sum + val, current_mask | (1 << y), dest_dict)
            
        gen_subsets(left, 0, 0, 0, left_sums)
        
        found = []
        
        # Iterate through all subsets of right part
        def check_right(items, start_idx, current_sum, current_mask):
            if start_idx == len(items):
                # Check if target - current_sum exists in left_sums
                rem = target - current_sum
                if rem in left_sums:
                    for l_mask in left_sums[rem]:
                        full_mask = l_mask | current_mask
                        # Ensure full_mask is not empty (though if target>0, it won't be)
                        if full_mask != 0:
                            found.append(full_mask)
                return
            
            # Skip
            check_right(items, start_idx + 1, current_sum, current_mask)
            # Take
            y, val = items[start_idx]
            # Pruning: if current_sum + val > target, we can't take it?
            # No, because left part can have negative? No, values are positive.
            # If current_sum + val > target, then rem < 0, so no match in left (sums are >= 0).
            # So we can prune.
            if current_sum + val <= target:
                check_right(items, start_idx + 1, current_sum + val, current_mask | (1 << y))
            else:
                # Even if we take it, sum exceeds target.
                # But we still need to check the "Skip" branch which we did above.
                # The "Take" branch is invalid.
                pass
                
        check_right(right, 0, 0, 0)
        
        # Deduplicate found masks
        # It's possible to find the same mask multiple times?
        # No, each subset of the full set is unique.
        # But wait, `found` might contain duplicates if the same sum can be formed in multiple ways?
        # No, the mask uniquely identifies the subset.
        # However, the same mask might be added multiple times if there are duplicate sums in left_sums?
        # `left_sums` is a dict: sum -> list of masks.
        # Each mask is unique in the list for a specific sum?
        # Yes, because each subset of `left` is unique.
        # So `found` contains unique masks.
        
        conflicts[x] = found
        
    return conflicts

def _is_valid_fast(S, conflicts):
    """
    S is a set of integers.
    conflicts[x] is a list of bitmasks M such that sum(1/y for y in M) == 1/x.
    """
    mask_S = 0
    for x in S:
        mask_S |= (1 << x)
        
    for x in S:
        for M in conflicts[x]:
            # Check if M is a subset of S
            if (M & mask_S) == M:
                return False
    return True

def construct():
    start = time.time()
    deadline = start + 85 # Safety margin
    
    # Precompute conflicts
    conflicts = _precompute_conflicts()
    
    # Initial solution: Greedy from largest to smallest
    best_set = set(range(13, 25))
    # Verify initial set
    if not _is_valid_fast(best_set, conflicts):
        # Fallback to a smaller valid set
        best_set = set()
        for c in range(24, 0, -1):
            trial = best_set | {c}
            if _is_valid_fast(trial, conflicts):
                best_set = trial

    best_size = len(best_set)
    
    # Local Search
    elements = list(range(1, 25))
    
    while time.time() < deadline:
        # Try to improve current best_set
        # 1. Try to add an element
        improved = False
        random.shuffle(elements)
        current_set = set(best_set)
        
        for x in elements:
            if x in current_set:
                continue
            trial = current_set | {x}
            if _is_valid_fast(trial, conflicts):
                current_set = trial
                improved = True
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
        
        # 2. Try swaps? Or just random restarts?
        # If no improvement, try a random restart or perturbation
        if not improved:
            # Perturb: remove a random element and try to fill
            if best_set:
                # Remove 1 or 2 random elements
                num_removals = random.randint(1, 2)
                for _ in range(num_removals):
                    if current_set:
                        victim = random.choice(list(current_set))
                        current_set.remove(victim)
                
                # Refill greedily
                random.shuffle(elements)
                for x in elements:
                    if x in current_set:
                        continue
                    trial = current_set | {x}
                    if _is_valid_fast(trial, conflicts):
                        current_set = trial
                
                if len(current_set) > best_size:
                    best_set = current_set
                    best_size = len(current_set)
            else:
                break
                
        # Check time
        if time.time() > deadline - 5:
            break

    return sorted(best_set)
