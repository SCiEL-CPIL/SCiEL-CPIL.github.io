"""Sidon set containing 0 with differences 1..30, minimizing max element."""
import random
import time

# EVOLVE-BLOCK-START
import time as _time
import bisect

def construct():
    deadline = _time.time() + 140  # safety margin below 150s
    random.seed(12345)

    # Fast Sidon check helpers
    # s: sorted list of elements
    # diffs: set of all positive differences between pairs in s
    
    def can_add(x, s, diffs):
        """Check if x can be added to s without violating Sidon property."""
        for y in s:
            d = abs(x - y)
            if d in diffs:
                return False
        return True

    def add_element(x, s, diffs):
        """Add x to s and update diffs. Assumes can_add is true."""
        for y in s:
            diffs.add(abs(x - y))
        # Insert x into sorted s
        idx = bisect.bisect_left(s, x)
        s.insert(idx, x)

    def missing_diffs(diffs):
        return [i for i in range(1, 31) if i not in diffs]

    def full_check(s):
        """Check if s is Sidon and covers 1..30. Returns (is_valid, max_val)"""
        if 0 not in s:
            return False, float('inf')
        diffs = set()
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                d = s[j] - s[i]
                if d in diffs:
                    return False, float('inf')
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False, float('inf')
        return True, max(s)

    # Greedy construction
    s = [0]
    diffs = set()
    
    max_iters = 100
    for _ in range(max_iters):
        if _time.time() > deadline - 5:
            break
            
        miss = missing_diffs(diffs)
        if not miss:
            break
            
        best_x = None
        best_max = float('inf')
        
        # Try to find an x that covers at least one missing difference
        # and results in the smallest possible new max(s)
        # Candidates come from y +/- d for y in s, d in miss
        candidates = set()
        for y in s:
            for d in miss:
                candidates.add(y + d)
                candidates.add(y - d)
        
        # Filter candidates: must be non-negative, not in s
        valid_candidates = [x for x in candidates if x >= 0 and x not in s]
        
        for x in valid_candidates:
            if can_add(x, s, diffs):
                new_max = max(max(s), x)
                if new_max < best_max:
                    best_max = new_max
                    best_x = x
                    
        if best_x is not None:
            add_element(best_x, s, diffs)
        else:
            # No candidate from missing diffs works. 
            # Try to add the smallest valid x > max(s) to expand the set
            start = max(s) + 1
            found = False
            # Limit search for new x to avoid infinite loop if stuck
            for x in range(start, start + 500):
                if can_add(x, s, diffs):
                    add_element(x, s, diffs)
                    found = True
                    break
            if not found:
                # Fallback: random search for a valid x
                for _ in range(5000):
                    x = random.randint(max(s) + 1, max(s) + 5000)
                    if can_add(x, s, diffs):
                        add_element(x, s, diffs)
                        found = True
                        break
                if not found:
                    break # Stuck

    # Local Search to minimize max element
    # We can try to replace an element with a smaller one if it doesn't break Sidon
    # and still covers all diffs.
    
    best_s = s[:]
    best_max = max(best_s)
    
    # Verify current solution
    valid, m = full_check(best_s)
    if not valid:
        # If greedy failed, try a simple random restart
        for _ in range(10):
            s = [0]
            diffs = set()
            for _ in range(max_iters):
                if _time.time() > deadline - 5:
                    break
                miss = missing_diffs(diffs)
                if not miss:
                    break
                best_x = None
                best_max_cand = float('inf')
                candidates = set()
                for y in s:
                    for d in miss:
                        candidates.add(y + d)
                        candidates.add(y - d)
                valid_candidates = [x for x in candidates if x >= 0 and x not in s]
                for x in valid_candidates:
                    if can_add(x, s, diffs):
                        new_max = max(max(s), x)
                        if new_max < best_max_cand:
                            best_max_cand = new_max
                            best_x = x
                if best_x is not None:
                    add_element(best_x, s, diffs)
                else:
                    start = max(s) + 1
                    found = False
                    for x in range(start, start + 500):
                        if can_add(x, s, diffs):
                            add_element(x, s, diffs)
                            found = True
                            break
                    if not found:
                        break
            valid, m = full_check(s)
            if valid and m < best_max:
                best_s = s
                best_max = m

    # Final Local Search Polish
    for _ in range(2000):
        if _time.time() > deadline - 2:
            break
            
        idx = random.randrange(len(best_s))
        if best_s[idx] == 0:
            continue
            
        old_val = best_s[idx]
        # Try a few smaller values
        for _ in range(5):
            new_val = random.randint(0, old_val - 1) if old_val > 0 else 0
            if new_val == old_val or new_val in best_s:
                continue
                
            trial = best_s[:]
            trial[idx] = new_val
            trial.sort()
            
            # Check Sidon and Coverage
            valid, m = full_check(trial)
            if valid and m < best_max:
                best_s = trial
                best_max = m
                break

    return best_s
# EVOLVE-BLOCK-END
