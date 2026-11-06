import time
import random

def _is_valid_set(S):
    """
    Checks if no element in S is the mean of a subset of at least 2 other elements.
    Uses bitset DP for subset sums.
    """
    n = len(S)
    if n < 3:
        return True
        
    for i, x in enumerate(S):
        others = [S[j] for j in range(n) if j != i]
        m = len(others)
        if m < 2:
            continue
            
        # dp[k] is an integer bitmask where bit s is 1 if sum s is achievable with k elements
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for val in others:
            # Iterate backwards to ensure each element is used at most once
            for k in range(m - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << val)
                    
        # Check if any k >= 2 has a subset summing to k * x
        # Max possible sum is sum(others)
        max_sum = sum(others)
        for k in range(2, m + 1):
            target = k * x
            if target > max_sum:
                break
            if (dp[k] >> target) & 1:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    # Pre-calculate the universe
    universe = list(range(1, 513))
    
    # We will run multiple restarts
    # Each restart builds a set and then improves it with local search
    
    while time.time() < deadline:
        # Phase 1: Build an initial valid set
        current_set = []
        # Shuffle universe for random picking
        candidates = universe[:]
        rng.shuffle(candidates)
        
        # Try to add as many as possible in one pass
        for x in candidates:
            if time.time() > deadline:
                break
            if len(current_set) >= 48:
                break
            new_set = current_set + [x]
            # Quick heuristic check: avoid 3-APs early to speed up
            # But full check is safer. The DP check is fast enough for small sets.
            if _is_valid_set(new_set):
                current_set = new_set
                
        if not current_set:
            current_set = [1]
            
        # Phase 2: Local Search to improve size
        stuck_count = 0
        max_stuck = 200
        
        while time.time() < deadline and stuck_count < max_stuck:
            improved = False
            
            # Try to add a random element
            available = [x for x in universe if x not in current_set]
            if available:
                # Randomly try a few additions
                for _ in range(5):
                    if not available:
                        break
                    idx = rng.randrange(len(available))
                    x = available[idx]
                    new_set = current_set + [x]
                    if _is_valid_set(new_set):
                        current_set = new_set
                        improved = True
                        available = [x for x in available if x != current_set[-1]]
                        break
            
            # If no addition, try a swap
            if not improved and len(current_set) > 0:
                # Try to swap a random element with a random new one
                for _ in range(5):
                    if not available:
                        break
                    old_idx = rng.randrange(len(current_set))
                    old_val = current_set[old_idx]
                    x = available[rng.randrange(len(available))]
                    
                    new_set = current_set[:old_idx] + current_set[old_idx+1:] + [x]
                    if _is_valid_set(new_set):
                        current_set = new_set
                        improved = True
                        # Update available list is expensive, just recalc or ignore for few swaps
                        # For simplicity, we recalc available if we swap often, 
                        # but here we just break and let the next loop iteration handle it
                        break
            
            if improved:
                stuck_count = 0
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    if best_size == 48:
                        return best_set
            else:
                stuck_count += 1
                # If stuck, remove a random element to create space
                if len(current_set) > 5:
                    remove_idx = rng.randrange(len(current_set))
                    current_set.pop(remove_idx)

    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    if len(s) > 0:
        print(f"Valid: {_is_valid_set(s)}")
