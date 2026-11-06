import time
import random
import sys

def construct():
    N = 512
    start_time = time.time()
    TIME_LIMIT = 55.0 # Safety margin for 60s limit
    
    best_set = set()
    best_size = 0
    
    # Helper to check if x can be added to S
    # Returns True if adding x creates no 4-AP
    def can_add(x, S):
        # Check if x is already in S (shouldn't happen)
        if x in S:
            return False
        
        # x can be part of a 4-AP in 4 positions:
        # 1. x, x+d, x+2d, x+3d  => need x+d, x+2d, x+3d in S
        # 2. x-d, x, x+d, x+2d  => need x-d, x+d, x+2d in S
        # 3. x-2d, x-d, x, x+d  => need x-2d, x-d, x+d in S
        # 4. x-3d, x-2d, x-d, x => need x-3d, x-2d, x-d in S
        
        # Max d such that points are in [1, N]
        # For case 1: x+3d <= N => d <= (N-x)//3
        # For case 4: x-3d >= 1 => d <= (x-1)//3
        
        max_d = (N - 1) // 3
        for d in range(1, max_d + 1):
            # Case 1: x is first
            if x + d <= N:
                if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                    return False
            else:
                break # d is too large for case 1, and larger d won't work for case 1
            
            # Case 2: x is second
            if x - d >= 1 and x + 2*d <= N:
                if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                    return False
            
            # Case 3: x is third
            if x - 2*d >= 1 and x + d <= N:
                if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                    return False
            
            # Case 4: x is fourth
            if x - 3*d >= 1:
                if (x - 3*d) in S and (x - 2*d) in S and (x - d) in S:
                    return False
            else:
                # If x-3d < 1, then for larger d, x-3d will be even smaller, 
                # but we still need to check smaller d's? 
                # Wait, the loop goes d=1, 2, ...
                # If x-3d < 1 for current d, it will be < 1 for all larger d.
                # But we might have missed case 1,2,3 for this d? No, they have their own bounds.
                # Actually, if x-3d < 1, case 4 is impossible.
                # We can't break the whole loop yet because case 1 might still be possible?
                # Case 1 requires x+3d <= N. If d is small, x+3d is small.
                # So we can't break based on case 4 condition alone.
                pass

        return True

    def greedy_randomized():
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Try to add any element
            # Iterate through candidates not in S
            # To make it faster, we can just check all 1..N not in S
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
                        # Don't break, keep trying to add more
            # If we couldn't add anything, try swaps
            if not improved:
                # Try 1-remove, 1-add
                # This is expensive: O(|S| * N * Cost(can_add))
                # Cost(can_add) is ~170 iterations.
                # |S| is ~100. N is 512.
                # 100 * 512 * 170 = 8.6M ops. Might be too slow for one step in Python.
                # We can limit the number of swaps tried or sample them.
                
                # Alternative: Random Kick
                # Remove a few random elements, then greedily add back.
                # This is often effective for escaping local optima.
                
                # Let's try a limited swap search
                # Pick a random subset of S to try removing
                removal_candidates = list(S)
                random.shuffle(removal_candidates)
                swaps_tried = 0
                for y in removal_candidates:
                    if swaps_tried > 50: # Limit work per local search step
                        break
                    S.remove(y)
                    added_any = False
                    # Try to add back x
                    # We can try to add y back first? No, that's useless.
                    # Try to add other x's
                    # To speed up, we can just try to add ALL valid x's greedily again?
                    # That might be expensive.
                    
                    # Let's just try to add one specific x that was previously blocked by y?
                    # We don't track that.
                    
                    # Simple approach: Try to add any x not in S (including y? no, y is out)
                    # If we can add at least 1, it's a neutral move (size same).
                    # If we can add 2, it's an improvement.
                    
                    # Let's just do a greedy fill on the modified set
                    # But that's O(N * Cost).
                    
                    # Let's try a specific heuristic:
                    # Remove y. Check if we can add any x.
                    # If yes, add it. If we added 1, size is same. 
                    # To ensure progress, we should only accept if we can add MORE than we removed?
                    # Or if we are stuck, accept neutral moves to change state.
                    
                    # Let's try: Remove y. Try to add x. If can add x, do it.
                    # Then check if we can add ANOTHER x2. If yes, we have a gain.
                    # If not, we have a neutral move.
                    
                    # To keep it simple and fast:
                    # Just do a greedy pass on the set after removing y.
                    # But that's expensive.
                    
                    # Let's stick to: Remove y, try to add ONE x.
                    # If added, accept. (Size unchanged, but state changed).
                    # This allows escaping local optima where no single add is possible,
                    # but a swap is.
                    
                    # Optimization: Only try swaps if we've been stuck for a while?
                    # For now, let's try a few random swaps.
                    
                    added = False
                    # Sample some x's to try adding
                    # Instead of all N, try a random subset
                    sample_size = 20
                    candidates_to_add = [x for x in range(1, N+1) if x not in S]
                    random.shuffle(candidates_to_add)
                    for x in candidates_to_add[:sample_size]:
                        if can_add(x, S):
                            S.add(x)
                            added = True
                            break # Just add one for now
                    
                    if not added:
                        # Restore y
                        S.add(y)
                    else:
                        # We removed y, added x. Size is same.
                        # Mark as improved to continue loop
                        improved = True
                        swaps_tried += 1
                        break # Restart the "try to add" phase with new state
        
        return S

    # Main Loop
    # 1. Get a good initial set
    # Try a few random greedy inits
    for i in range(10):
        S = greedy_randomized()
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
        # Check time
        if time.time() - start_time > TIME_LIMIT * 0.1:
            break

    # 2. Local Search on the best set
    current_set = set(best_set)
    
    while time.time() - start_time < TIME_LIMIT:
        current_set = local_search(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        # If we are stuck (no improvement in last few iterations), do a kick
        # Simple kick: Remove 5 random elements, then greedily add back
        # This helps escape deep local minima
        if random.random() < 0.1: # 10% chance to kick
            S_kick = set(current_set)
            if len(S_kick) > 5:
                to_remove = random.sample(list(S_kick), 5)
                for y in to_remove:
                    S_kick.discard(y)
                
                # Greedy fill
                candidates = [x for x in range(1, N+1) if x not in S_kick]
                random.shuffle(candidates)
                for x in candidates:
                    if can_add(x, S_kick):
                        S_kick.add(x)
                
                if len(S_kick) > len(current_set):
                    current_set = S_kick
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)

    return sorted(best_set)

if __name__ == "__main__":
    # Test
    res = construct()
    print(len(res))
