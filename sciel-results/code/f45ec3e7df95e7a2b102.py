"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin before 90s limit (evaluator kills at 90s)
    
    n = 256
    # We work with indices 0..255 representing numbers 1..256
    # num = idx + 1
    
    # Precompute conflicts for each number x (1..256)
    # conflict_mask[x] is a bitmask of numbers that cannot be in the set with x
    # Actually, for 3-APs, x conflicts with y if they are part of a 3-AP.
    # It's easier to check validity on the fly using the boolean array.
    
    # 1. Greedy Initialization
    # Sort numbers by 1/x descending (i.e., x ascending) to maximize reciprocal sum initially.
    # However, a simple greedy by x ascending might block many larger numbers.
    # A better heuristic for 3-AP free sets is often to pick numbers based on binary representation 
    # (Salem-Spencer type sets) or just simple greedy with lookahead.
    # Let's stick to a robust greedy: try to add x if it doesn't create a 3-AP.
    # To improve on simple ascending greedy, we can try multiple random orderings or 
    # a "weighted" greedy.
    
    # Let's use a standard greedy first to get a valid baseline.
    # Then we will use local search to improve it.
    
    def is_valid_addition(x, S_bool):
        # x is 1-based
        # Check if adding x creates a 3-AP with existing elements in S_bool
        # A 3-AP is (a, b, c) with a+c=2b.
        # If we add x, x can be:
        # 1. Middle: a = x-d, c = x+d. Need S_bool[a-1] and S_bool[c-1] to be True.
        # 2. End: a = x-2d, b = x-d. Need S_bool[a-1] and S_bool[b-1] to be True.
        # 3. Start: b = x+d, c = x+2d. Need S_bool[b-1] and S_bool[c-1] to be True.
        
        # Max d for middle: x-1 >= 1 => d <= x-1. Also x+d <= 256 => d <= 256-x.
        max_d_mid = min(x - 1, 256 - x)
        for d in range(1, max_d_mid + 1):
            a = x - d
            c = x + d
            if S_bool[a-1] and S_bool[c-1]:
                return False
        
        # Max d for end: x-2d >= 1 => d <= (x-1)/2.
        max_d_end = (x - 1) // 2
        for d in range(1, max_d_end + 1):
            a = x - 2*d
            b = x - d
            if S_bool[a-1] and S_bool[b-1]:
                return False
        
        # Max d for start: x+2d <= 256 => d <= (256-x)/2.
        max_d_start = (256 - x) // 2
        for d in range(1, max_d_start + 1):
            b = x + d
            c = x + 2*d
            if S_bool[b-1] and S_bool[c-1]:
                return False
        
        return True

    def get_score(S_bool):
        # Sum of 1/x for x in S
        # S_bool[i] is True if i+1 is in S
        # Create array of 1/(i+1)
        # This is called frequently, so precompute reciprocals
        return float(np.sum(recips[S_bool]))

    # Precompute reciprocals
    recips = 1.0 / np.arange(1, n + 1)
    
    # Generate a few initial candidates using different greedy strategies
    candidates = []
    
    # Strategy 1: Standard Greedy (Ascending)
    S1 = np.zeros(n, dtype=bool)
    for x in range(1, n + 1):
        if is_valid_addition(x, S1):
            S1[x-1] = True
    candidates.append(S1)
    
    # Strategy 2: Greedy with Random Tie-breaking/Ordering (Perturbed)
    # Try a few random shuffles of the "priority" of adding numbers?
    # Or simply: Greedy but skip some small numbers to allow more medium ones?
    # Let's try a "Lookahead" greedy: 
    # Actually, a known good heuristic for 3-AP free sets is to take numbers 
    # with no '1' in their base-3 representation (Salem-Spencer).
    # For n=256, let's generate that set.
    S2 = np.zeros(n, dtype=bool)
    for x in range(1, n + 1):
        # Check if base 3 representation has only 0s and 1s
        temp = x
        valid = True
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            S2[x-1] = True
    candidates.append(S2)
    
    # Strategy 3: Randomized Greedy
    # Pick numbers in random order, add if valid.
    # This can sometimes find better structures.
    for _ in range(5):
        S3 = np.zeros(n, dtype=bool)
        order = list(range(1, n + 1))
        random.shuffle(order)
        # Bias towards smaller numbers? 
        # Let's sort by 1/x with some noise
        weights = np.array([1.0/x for x in range(1, n+1)])
        noise = np.random.exponential(0.1, size=n)
        scores = weights + noise
        order_idx = np.argsort(-scores) + 1 # 1-based
        for x in order_idx:
            if is_valid_addition(int(x), S3):
                S3[int(x)-1] = True
        candidates.append(S3)

    # Select best candidate
    best_S = None
    best_score = -1
    for S in candidates:
        sc = get_score(S)
        if sc > best_score:
            best_score = sc
            best_S = S.copy()
    
    current_S = best_S
    current_score = best_score

    # 2. Local Search Phase
    # Moves:
    # A. Add x (if valid) -> Score increases. Always accept.
    # B. Remove x -> Score decreases. Accept if it allows adding a better set of numbers? 
    #    Simple hill climbing on "Remove" won't work because it lowers score.
    #    We need "Swap" or "Remove-Add" sequences.
    
    # Let's implement a local search that tries:
    # 1. Try to add any valid x. If found, add it. (Hill Climb)
    # 2. If stuck (no valid additions), try removing one element x.
    #    Then try to add any valid y.
    #    If the net change in score (1/y - 1/x) is positive, accept the swap (remove x, add y).
    #    If not, try adding multiple y's? That's complex.
    #    Let's stick to single swaps first.
    
    # Optimization: Precompute potential additions for a given state?
    # Too expensive to recompute fully each time.
    
    # Let's iterate until no improvement or time runs out.
    no_improvement_count = 0
    max_no_improvement = 100 # Reset counter if we find a big improvement?
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Phase A: Try to add any element
        # Iterate through all x not in S
        # If valid, add it.
        # To be efficient, we can check all x at once? No, validity depends on S.
        # We can just loop x from 1 to n.
        added_any = False
        for x in range(1, n + 1):
            if not current_S[x-1]:
                if is_valid_addition(x, current_S):
                    current_S[x-1] = True
                    current_score += 1.0 / x
                    added_any = True
                    improved = True
                    # After adding, we might be able to add more, but the loop continues.
                    # However, adding x might invalidate previous "invalid" checks? 
                    # No, adding elements only makes it *harder* to add more.
                    # So if x was invalid before, it remains invalid.
                    # We can just continue the loop.
        
        if added_any:
            no_improvement_count = 0
            continue # Restart the scan as the set changed
        
        # Phase B: Try Swaps (Remove x, Add y)
        # For each x in S, remove it, then see if we can add any y > x (in terms of value 1/y < 1/x? No, we want to gain score)
        # Actually, we want to remove x and add y such that 1/y > 1/x is impossible if y<x.
        # We want to remove a "blocking" element.
        # Let's iterate x in S.
        # Remove x.
        # Find all valid y not in S.
        # If any y exists, calculate delta = 1/y - 1/x.
        # If delta > 0, accept.
        # If delta <= 0, we might still accept if it unlocks more additions? 
        # For now, strict hill climbing on swap gain.
        
        swap_found = False
        # To speed up, we can iterate x in order of "removability"? 
        # Just iterate all.
        for x in range(1, n + 1):
            if current_S[x-1]:
                # Temporarily remove x
                current_S[x-1] = False
                temp_score = current_score - 1.0 / x
                
                # Try to find a y to add
                best_y = -1
                best_gain = 0
                for y in range(1, n + 1):
                    if not current_S[y-1]:
                        if is_valid_addition(y, current_S):
                            gain = 1.0 / y - 1.0 / x
                            if gain > best_gain:
                                best_gain = gain
                                best_y = y
                
                if best_y != -1 and best_gain > 0:
                    # Accept swap
                    current_S[best_y-1] = True
                    current_score = temp_score + 1.0 / best_y
                    improved = True
                    swap_found = True
                    # Break out to restart search with new state
                    break
                else:
                    # Restore x
                    current_S[x-1] = True
        
        if swap_found:
            no_improvement_count = 0
            continue
        
        # Phase C: Try "Remove and Add Multiple" or "2-Swaps"?
        # If no single swap improves, we are at a local optimum for single swaps.
        # We can try removing 2 elements and adding 2? 
        # Or just restart with a perturbed state?
        
        # Let's try a "Kick": Remove a random element, then greedily add back as many as possible.
        # This can escape deep local minima.
        
        if no_improvement_count > 20:
            # Perturb
            # Pick a random element to remove
            if current_score > 0:
                in_set = np.where(current_S)[0] + 1
                if len(in_set) > 0:
                    x_remove = in_set[random.randint(0, len(in_set)-1)]
                    current_S[x_remove-1] = False
                    current_score -= 1.0 / x_remove
                    
                    # Now greedily add back
                    for x in range(1, n + 1):
                        if not current_S[x-1]:
                            if is_valid_addition(x, current_S):
                                current_S[x-1] = True
                                current_score += 1.0 / x
                    no_improvement_count = 0
                    continue
            else:
                break

        no_improvement_count += 1
        if no_improvement_count > 500:
            # Stuck for a long time, maybe restart entirely?
            # Let's just break and return best found so far.
            # Or try a different initial candidate?
            break

    # Ensure we return a list of integers
    result = [int(i) + 1 for i in np.where(current_S)[0]]
    
    # Final check: is it non-empty?
    if not result:
        # Fallback: just return {1}
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
