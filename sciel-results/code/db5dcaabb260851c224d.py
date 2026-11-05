import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Precompute max d for 4-APs and 3-APs
    max_d_4 = N // 3
    max_d_3 = N // 2
    
    def count_3aps_with(x, in_set):
        """Count 3-APs in the set defined by in_set that include x."""
        count = 0
        # x is middle: x-d, x, x+d
        for d in range(1, max_d_3 + 1):
            if x - d >= 1 and x + d <= N:
                if in_set[x - d] and in_set[x + d]:
                    count += 1
        # x is first: x, x+d, x+2d
        for d in range(1, max_d_3 + 1):
            if x + 2*d <= N:
                if in_set[x + d] and in_set[x + 2*d]:
                    count += 1
        # x is last: x-2d, x-d, x
        for d in range(1, max_d_3 + 1):
            if x - 2*d >= 1:
                if in_set[x - 2*d] and in_set[x - d]:
                    count += 1
        return count

    def creates_4ap(x, in_set):
        """Check if adding x to set defined by in_set creates a 4-AP."""
        # x is first: x, x+d, x+2d, x+3d
        for d in range(1, max_d_4 + 1):
            if x + 3*d <= N:
                if in_set[x + d] and in_set[x + 2*d] and in_set[x + 3*d]:
                    return True
        # x is second: x-d, x, x+d, x+2d
        for d in range(1, max_d_4 + 1):
            if x - d >= 1 and x + 2*d <= N:
                if in_set[x - d] and in_set[x + d] and in_set[x + 2*d]:
                    return True
        # x is third: x-2d, x-d, x, x+d
        for d in range(1, max_d_4 + 1):
            if x - 2*d >= 1 and x + d <= N:
                if in_set[x - 2*d] and in_set[x - d] and in_set[x + d]:
                    return True
        # x is fourth: x-3d, x-2d, x-d, x
        for d in range(1, max_d_4 + 1):
            if x - 3*d >= 1:
                if in_set[x - 3*d] and in_set[x - 2*d] and in_set[x - d]:
                    return True
        return False

    def total_3aps(in_set):
        # Sum count_3aps_with for all x in set, then divide by 3 
        # because each 3-AP is counted 3 times (once for each member)
        total = 0
        for i in range(1, N + 1):
            if in_set[i]:
                total += count_3aps_with(i, in_set)
        return total // 3

    def random_valid_set():
        in_set = [False] * (N + 1)
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        for x in elements:
            if not creates_4ap(x, in_set):
                in_set[x] = True
        return in_set

    best_score = -1
    best_in_set = [False] * (N + 1)
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        in_set = random_valid_set()
        current_score = total_3aps(in_set)
        
        if current_score > best_score:
            best_score = current_score
            best_in_set = in_set[:]
            
        # Simulated Annealing
        T = 50.0
        T_min = 0.01
        cooling = 0.995
        
        # Estimate steps based on remaining time
        # Each step involves ~200-400 ops in Python. 
        # Let's aim for a fixed number of steps per restart, say 2000.
        steps = 2000
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Choose a move: Add, Remove, or Swap
            move_type = random.random()
            
            if move_type < 0.4: # Add
                # Pick a random element not in set
                candidates = [i for i in range(1, N + 1) if not in_set[i]]
                if not candidates: continue
                x = random.choice(candidates)
                if creates_4ap(x, in_set):
                    continue # Reject if invalid
                delta = count_3aps_with(x, in_set)
                # Acceptance
                if delta > 0 or random.random() < math.exp(delta / T):
                    in_set[x] = True
                    current_score += delta
            elif move_type < 0.8: # Remove
                # Pick a random element in set
                candidates = [i for i in range(1, N + 1) if in_set[i]]
                if not candidates: continue
                x = random.choice(candidates)
                delta = -count_3aps_with(x, in_set)
                # Removing always keeps it valid (removing can't create 4-AP)
                if delta > 0 or random.random() < math.exp(delta / T):
                    in_set[x] = False
                    current_score += delta
            else: # Swap
                # Pick one in, one out
                in_candidates = [i for i in range(1, N + 1) if in_set[i]]
                out_candidates = [i for i in range(1, N + 1) if not in_set[i]]
                if not in_candidates or not out_candidates: continue
                x_out = random.choice(in_candidates)
                x_in = random.choice(out_candidates)
                
                # Check if swapping is valid
                # Temporarily remove x_out
                in_set[x_out] = False
                # Check if x_in creates 4-AP
                if creates_4ap(x_in, in_set):
                    in_set[x_out] = True
                    continue
                # Calculate delta
                # Remove x_out contribution, Add x_in contribution
                # Note: count_3aps_with depends on current set.
                # When we remove x_out, some 3-APs involving x_out disappear.
                # When we add x_in, some 3-APs involving x_in appear.
                # But there might be 3-APs involving BOTH x_out and x_in?
                # A 3-AP has 3 elements. It can't contain both x_out and x_in 
                # unless x_out and x_in are in the same 3-AP.
                # If they are in the same 3-AP, removing x_out destroys it, 
                # adding x_in doesn't recreate it (x_in is different).
                # So delta = -count_3aps_with(x_out, set_with_x_out) + count_3aps_with(x_in, set_without_x_out)
                # We need count_3aps_with(x_out) BEFORE removal.
                # We need count_3aps_with(x_in) AFTER removal.
                
                # Let's re-calculate carefully.
                # Current state: in_set has x_out, not x_in.
                # Score S.
                # Remove x_out: S' = S - count_3aps_with(x_out, in_set)
                # Add x_in: S'' = S' + count_3aps_with(x_in, in_set_without_x_out)
                # Delta = S'' - S = -count_3aps_with(x_out, in_set) + count_3aps_with(x_in, in_set_without_x_out)
                
                # We already set in_set[x_out] = False.
                # So in_set is currently "without x_out".
                # But we need count for x_out in the ORIGINAL set.
                # Let's just compute the new score directly if the move is small?
                # No, total_3aps is O(N^2). Delta is O(N).
                
                # We lost the original count for x_out. 
                # Let's just revert and try a simpler swap logic or re-evaluate.
                # Actually, just calculating total_3aps is safer but slower.
                # Given N=256, total_3aps is ~ 256 * 256 = 65k ops.
                # If we do this for every swap, it might be too slow for 2000 steps.
                # 2000 * 65k = 130M ops. Might be tight in Python.
                
                # Let's stick to Add/Remove mostly, or optimize Swap.
                # For now, reject Swap if we can't compute delta fast.
                # Or just use Add/Remove.
                in_set[x_out] = True
                continue

            # Update best
            if current_score > best_score:
                best_score = current_score
                best_in_set = in_set[:]
                
            T *= cooling
            if T < T_min:
                T = T_min

    # Convert best_in_set to sorted list
    result = [i for i in range(1, N + 1) if best_in_set[i]]
    return result

# Test/Validation
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}, Score: {total_3aps([False] + [x in res for x in range(1, 257)])}")
