"""New program for Sidon set with differences 1..30 using Constraint-Driven Backtracking and Greedy Shrink."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a Sidon set containing 0 with all differences 1..30.
    Minimize the largest element.
    
    Strategy:
    1. Use Constraint-Driven Backtracking (CSP) to find a valid Sidon set.
       - Use bitmasks for difference tracking for speed.
       - Use pruning based on maximum possible future differences.
       - Use heuristic ordering to find solutions quickly.
    2. Use a Greedy Shrink local search to minimize the largest element (diameter).
       - Iteratively attempt to move elements to smaller valid positions.
    """
    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in target_diffs:
        target_mask |= (1 << d)
    
    def is_valid_sidon(s):
        """Check if set s is a Sidon set."""
        diffs = set()
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(s[j] - s[i])
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_missing(s):
        """Get missing differences from 1..30 as a set."""
        diffs = set()
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(abs(s[j] - s[i]))
        return target_diffs - diffs

    def backtrack(current_s, used_mask, depth=0):
        """
        Recursively add elements to current_s until all target differences are covered.
        Returns a valid set or None.
        """
        if (used_mask & target_mask) == target_mask:
            return list(current_s)
        
        if len(current_s) >= 32:
            return None
            
        # Pruning: Check if it's even possible to cover remaining missing diffs
        missing_mask = target_mask & ~used_mask
        missing_count = bin(missing_mask).count("1")
        
        # Max new differences we can generate with remaining slots
        current_size = len(current_s)
        remaining_slots = 32 - current_size
        # New elements will form pairs with existing elements and each other
        # Max new diffs = remaining_slots * current_size + remaining_slots * (remaining_slots - 1) // 2
        max_new_diffs = remaining_slots * current_size + (remaining_slots * (remaining_slots - 1)) // 2
        
        if missing_count > max_new_diffs:
            return None

        # Generate candidates
        candidates = []
        # We will try x from 1 up to a limit.
        # Limit: max(current_s) + 30 is a reasonable upper bound for "compact" sets.
        # But initially, we might need to go higher if we make a bad choice.
        # Let's try 1..150 first to be safe.
        
        for x in range(1, 150):
            if x in current_s:
                continue
                
            # Calculate new differences
            new_diffs_mask = 0
            valid = True
            for s_i in current_s:
                d = abs(x - s_i)
                # If this difference is already used, it's not a Sidon set
                if (used_mask >> d) & 1:
                    valid = False
                    break
                new_diffs_mask |= (1 << d)
            
            if not valid:
                continue
                
            # Check if this x adds any new target differences
            new_target_diffs = new_diffs_mask & target_mask & ~used_mask
            if new_target_diffs == 0:
                continue # Doesn't help cover missing diffs
            
            # Score the candidate: prefer those covering more missing diffs
            # And prefer smaller x to keep diameter small
            score = bin(new_target_diffs).count("1") * 1000 - x
            candidates.append((score, x))
            
        # Sort by score (descending score)
        candidates.sort(key=lambda item: -item[0])
        
        for score, x in candidates:
            trial_s = sorted(current_s + [x])
            
            # Recalculate mask for x with current_s
            x_mask = 0
            for s_i in current_s:
                d = abs(x - s_i)
                x_mask |= (1 << d)
                
            res = backtrack(trial_s, used_mask | x_mask, depth + 1)
            if res:
                return res
                
        return None

    # 1. Find initial solution using CSP
    start_time = time.time()
    
    initial_s = backtrack([0], 0)
    
    if initial_s is None:
        # Fallback: Greedy construction
        s = [0]
        used_mask = 0
        while (used_mask & target_mask) != target_mask and len(s) < 32:
            best_x = -1
            best_score = -1
            for x in range(1, 300):
                if x in s: continue
                valid = True
                new_mask = 0
                for s_i in s:
                    d = abs(x - s_i)
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_mask |= (1 << d)
                if not valid: continue
                new_target = new_mask & target_mask & ~used_mask
                if new_target == 0: continue
                score = bin(new_target).count("1") * 1000 - x
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x == -1: break
            s.append(best_x)
            for s_i in s[:-1]:
                d = abs(best_x - s_i)
                used_mask |= (1 << d)
        if (used_mask & target_mask) == target_mask:
            initial_s = s
        else:
            initial_s = s # Return best effort

    if not initial_s:
        return [0]
        
    # Ensure validity
    if get_missing(initial_s) or not is_valid_sidon(initial_s):
        return sorted(initial_s)

    # 2. Greedy Shrink to minimize max element
    def greedy_shrink(s):
        s = sorted(s)
        improved = True
        while improved:
            improved = False
            # Try to shrink each element, starting from the largest
            for i in range(len(s) - 1, 0, -1):
                old_val = s[i]
                # Try values from 1 to old_val - 1
                for new_val in range(1, old_val):
                    if new_val in s:
                        continue
                    new_s = sorted([v for v in s if v != old_val] + [new_val])
                    if is_valid_sidon(new_s) and not get_missing(new_s):
                        s = new_s
                        improved = True
                        break # Restart the while loop
                if improved:
                    break
        return s

    current_s = greedy_shrink(initial_s)
    
    # 3. Randomized Restarts to escape local minima
    best_s = current_s
    best_max = max(current_s)
    
    # Try a few random perturbations and shrink again
    for _ in range(10):
        if time.time() - start_time > 110:
            break
            
        # Perturb: randomly move one element
        perturbed_s = list(current_s)
        idx = random.randint(1, len(perturbed_s) - 1)
        old_val = perturbed_s[idx]
        new_val = random.randint(1, max(1, max(perturbed_s)))
        if new_val not in perturbed_s:
            perturbed_s[idx] = new_val
            perturbed_s = sorted(perturbed_s)
            
            if is_valid_sidon(perturbed_s) and not get_missing(perturbed_s):
                shrunk_s = greedy_shrink(perturbed_s)
                if max(shrunk_s) < best_max:
                    best_s = shrunk_s
                    best_max = max(shrunk_s)
                    current_s = shrunk_s
                elif max(shrunk_s) == best_max:
                    # Keep current best, but maybe update current_s for diversity?
                    pass
            else:
                # If perturbation invalid, skip
                pass

    return sorted(best_s)
# EVOLVE-BLOCK-END
