import time
import sys

# Global variables
best_set = []
best_size = 0
start_time = 0
time_limit = 115.0

SUMS_SIZE = 570

def construct():
    global best_set, best_size, start_time
    
    start_time = time.time()
    
    initial_set = [1, 2, 5, 11]
    initial_sums = bytearray(SUMS_SIZE)
    
    for i in range(len(initial_set)):
        for j in range(i, len(initial_set)):
            s = initial_set[i] + initial_set[j]
            initial_sums[s] = 1
            
    # 1. Greedy Construction to find a good initial lower bound
    # Strategy: Add the largest valid candidate repeatedly.
    # This spreads marks out, which is good for Sidon sets.
    
    current_set = initial_set[:]
    current_sums = initial_sums[:]
    
    # Candidates are 12..284
    # We will maintain a list of available candidates
    # To make it fast, we can just iterate 284 down to 12 and check validity.
    # But we need to add them one by one.
    
    # Let's create a list of all potential candidates
    all_candidates = list(range(284, 11, -1)) # Descending order
    
    # We will build the greedy set
    greedy_set = initial_set[:]
    greedy_sums = initial_sums[:]
    
    # To check validity quickly, we might need to filter candidates.
    # But since N is small (284), O(N^2) checks are fine.
    
    for x in all_candidates:
        # Check if x can be added to greedy_set
        if 2 * x < SUMS_SIZE and greedy_sums[2 * x]:
            continue
        valid = True
        for y in greedy_set:
            s = x + y
            if s < SUMS_SIZE and greedy_sums[s]:
                valid = False
                break
        if valid:
            greedy_set.append(x)
            greedy_sums[2 * x] = 1
            for y in greedy_set:
                if y != x:
                    greedy_sums[x + y] = 1
                    
    # Update best
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set[:]
        
    # 2. DFS to improve
    # We will try to find sets of size best_size + 1, then + 2, etc.
    # This avoids wasting time on impossible large targets if smaller ones are also hard.
    # Actually, if best_size is 18, we try 19. If we find 19, we try 20.
    # If we don't find 19, we stop (since 20 implies 19).
    
    candidates = []
    for x in range(12, 285):
        # Filter candidates that are invalid against the INITIAL set?
        # No, we need them to be valid against the CURRENT set during DFS.
        # So we keep all candidates 12..284.
        candidates.append(x)
        
    # Sort candidates for DFS.
    # Descending is generally good for spreading.
    candidates.sort(reverse=True)
    
    K = len(candidates)
    
    def dfs(current_set, current_sums, avail_mask, target_size):
        global best_set, best_size
        
        if time.time() - start_time > time_limit:
            return False
            
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        if current_size == target_size:
            return True
            
        # Pruning: Upper bound
        remaining = avail_mask.bit_count() if hasattr(avail_mask, 'bit_count') else bin(avail_mask).count('1')
        if current_size + remaining < target_size:
            return False
            
        # Iterate through available candidates
        mask = avail_mask
        while mask:
            lsb = mask & -mask
            i = lsb.bit_length() - 1
            mask ^= lsb
            
            x = candidates[i]
            
            # Check if x can be added
            if 2 * x < SUMS_SIZE and current_sums[2 * x]:
                continue
                
            can_add = True
            for y in current_set:
                s = x + y
                if s < SUMS_SIZE and current_sums[s]:
                    can_add = False
                    break
                    
            if not can_add:
                continue
                
            # Add x
            current_set.append(x)
            current_sums[2 * x] = 1
            for y in current_set:
                if y != x:
                    current_sums[x + y] = 1
            
            # Compute new_avail_mask
            new_mask = avail_mask & ~lsb
            
            # Filter out candidates that conflict with x given current_set
            temp_mask = new_mask
            filtered_mask = 0
            while temp_mask:
                lsb_j = temp_mask & -temp_mask
                j = lsb_j.bit_length() - 1
                temp_mask ^= lsb_j
                
                v = candidates[j]
                
                if 2 * v < SUMS_SIZE and current_sums[2 * v]:
                    continue
                valid_v = True
                for y in current_set:
                    s = v + y
                    if s < SUMS_SIZE and current_sums[s]:
                        valid_v = False
                        break
                if valid_v:
                    filtered_mask |= lsb_j
            
            if dfs(current_set, current_sums, filtered_mask, target_size):
                return True
                
            # Backtrack
            current_set.pop()
            current_sums[2 * x] = 0
            for y in current_set:
                current_sums[x + y] = 0
                
        return False

    # Try to improve from best_size
    # We start from best_size + 1 up to 20
    # If we find a set of size T, we update best_size to T and try T+1.
    
    # Note: The DFS starts from the initial set {1,2,5,11}.
    # It does NOT start from the greedy set.
    # This is correct because the greedy set might be a local optimum or have a specific structure that hinders finding a larger set.
    # However, starting from scratch is expensive.
    # But with the pruning `current_size + remaining < target_size`, and starting from the bottom (size 4), it's fine.
    # The key is that we try small targets first.
    
    for target in range(best_size + 1, 21):
        if time.time() - start_time > time_limit - 5:
            break
            
        init_set_copy = initial_set[:]
        init_sums_copy = initial_sums[:]
        
        init_mask = (1 << K) - 1
        
        if dfs(init_set_copy, init_sums_copy, init_mask, target):
            # If we found a set of this size, loop continues to next target
            continue
        else:
            # If we didn't find a set of this size, and we exhausted the search (not just timeout),
            # then no larger set is possible?
            # Wait, if DFS returns False, it means it exhausted the search space for that target.
            # So if we can't find size T, we can't find size T+1.
            # So we should break.
            # BUT, if it returned False due to timeout, we shouldn't break, we should just stop trying larger targets.
            # However, the DFS returns False on timeout too.
            # We need to distinguish between "exhausted" and "timeout".
            # For simplicity, if it returns False, we break the loop.
            # This is safe: if we couldn't find T, we likely can't find T+1 efficiently.
            break
            
    return best_set
