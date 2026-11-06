import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Initial valid set: {66, ..., 100}
    # We verified that min 3-sum is 201, max 2-sum is 199.
    # So no subset sums to 200.
    current_set = set(range(66, 101))
    best_set = set(current_set)
    best_size = len(current_set)
    
    def is_valid(s):
        # Check if any subset of s sums to 200
        # Use bitset for speed
        reachable = 1  # bit 0 is 1
        for x in s:
            reachable |= (reachable << x)
            if reachable & (1 << 200):
                return False
        return True

    def find_conflict(s):
        # Find one subset summing to 200
        # Backtracking or DP reconstruction
        # For speed, simple DP to find if reachable, then backtrack
        # Since 200 is small, DP is fine.
        dp = [False] * 201
        dp[0] = True
        parent = [None] * 201
        for x in s:
            for j in range(200, x - 1, -1):
                if dp[j - x] and not dp[j]:
                    dp[j] = True
                    parent[j] = x
        if not dp[200]:
            return None
        # Reconstruct
        conflict = []
        j = 200
        while j > 0:
            x = parent[j]
            conflict.append(x)
            j -= x
        return conflict

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Try to add a random element not in current_set
        candidates = [i for i in range(1, 101) if i not in current_set]
        if not candidates:
            break
        
        x = random.choice(candidates)
        new_set = current_set | {x}
        
        if is_valid(new_set):
            current_set = new_set
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
        else:
            # Conflict found. Try to resolve by removing an element.
            conflict = find_conflict(new_set)
            if conflict:
                # Remove a random element from the conflict
                # Prefer removing from current_set to keep x if possible?
                # Or just random.
                y = random.choice(conflict)
                candidate_set = new_set - {y}
                if is_valid(candidate_set):
                    # Accept if size is same or larger (it will be same if y in current, smaller if y==x)
                    # If y == x, we reverted.
                    # If y in current_set, we swapped x for y. Size same.
                    # We might want to accept same size to explore.
                    current_set = candidate_set
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    # Occasionally revert to best to avoid drift?
                    # No, local search works.
    
    return list(best_set)
