import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (150s kill, 40s buffer)
    
    # Greedy initialization with scoring
    S = set()
    counts = {}
    
    # Helper to check if x can be added to current S with current counts
    def can_add(x):
        # Check 2*x
        if counts.get(2*x, 0) >= 2:
            return False
        # Check x + s for s in S
        for s in S:
            if counts.get(x + s, 0) >= 2:
                return False
        return True

    # Helper to add x
    def add(x):
        S.add(x)
        counts[2*x] = counts.get(2*x, 0) + 1
        for s in S:
            if s != x:
                counts[x + s] = counts.get(x + s, 0) + 1

    # Helper to remove x
    def remove(x):
        S.remove(x)
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for s in S:
            counts[x + s] -= 1
            if counts[x + s] == 0:
                del counts[x + s]

    # Initial greedy: pick smallest available
    for x in range(1, 1001):
        if can_add(x):
            add(x)
        if time.time() > deadline:
            break

    # Local search with targeted swaps
    candidates = [x for x in range(1, 1001) if x not in S]
    
    # Precompute for each candidate which elements in S block it
    # A candidate y is blocked by x in S if count(x+y) >= 2 or count(2y) >= 2
    # For swap: remove x from S, then check if y can be added to S\{x}
    
    best_size = len(S)
    
    while time.time() < deadline:
        # Phase 1: Try to add any candidate
        added_any = False
        for i in range(len(candidates)):
            y = candidates[i]
            if can_add(y):
                add(y)
                candidates.pop(i)
                added_any = True
                if len(S) > best_size:
                    best_size = len(S)
                break
        if added_any:
            continue
        
        # Phase 2: Try targeted swaps
        # For each candidate y, find which elements in S block it
        # Then try removing one blocker and adding y
        
        # Build blocker map: for each candidate y, list of x in S that block y
        # y is blocked by x if counts.get(x+y, 0) >= 2 (after removing x, this count decreases by 1)
        # Actually, if we remove x, then for any s in S\{x}, count(s+y) is unchanged.
        # The only sums involving x are x+s for s in S. So removing x doesn't change count(s+y) for s != x.
        # Wait, that's not right. counts[s+y] includes all pairs (a,b) in S with a+b=s+y.
        # If we remove x, then pairs involving x are removed. So count(s+y) might decrease if there was a pair (x, s+y-x) in S.
        # This is complex. Let's just do a direct check for swaps.
        
        # Simpler: try random swaps but be smarter about which ones to try
        if S and candidates:
            # Try a few random swaps per iteration
            attempts = 0
            max_attempts = min(50, len(candidates) * len(S) // 100)
            while attempts < max_attempts and time.time() < deadline:
                x = random.choice(list(S))
                y = random.choice(candidates)
                attempts += 1
                
                # Check if S\{x} U {y} is valid
                # We need: for all s in S, s != x: count(s+y) < 2 (in current counts, which include x)
                # But count(s+y) in current counts includes pairs involving x.
                # If we remove x, count(s+y) becomes count(s+y) - [number of pairs (x, s+y-x) in S with s+y-x != x]
                # This is getting complicated. Let's just temporarily remove and check.
                
                # Temporary removal approach:
                # Remove x, check can_add(y), if yes keep, else put x back.
                
                remove(x)
                if can_add(y):
                    add(y)
                    candidates.remove(y)
                    candidates.append(x)
                    if len(S) > best_size:
                        best_size = len(S)
                    break
                else:
                    add(x)
                    # x is back in S, y is still in candidates
                    # We need to make sure x is not in candidates (it shouldn't be)
        
        # Check if we're making progress
        if time.time() > deadline - 10:
            break
    
    return sorted(list(S))
