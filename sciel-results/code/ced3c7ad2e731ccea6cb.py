"""Rewritten program: Numpy-accelerated greedy and local search for divisibility-incomparable subset sums."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START

def _is_valid_numpy(nums):
    """Exact check using numpy for speed."""
    n = len(nums)
    if n == 0:
        return True
    
    # Generate subset sums
    sums = {0}
    for x in nums:
        sums = sums | {s + x for s in sums}
    sums.discard(0)
    
    if len(sums) < 2:
        return True
        
    ss = np.array(list(sums), dtype=np.int64)
    
    # Check divisibility
    # For each a in ss, check if any other b in ss is divisible by a.
    # Since elements are distinct, ss % a will have exactly one 0 (at a's index) if valid.
    # If invalid, it will have > 1 zeros.
    for i in range(len(ss)):
        a = ss[i]
        if (ss % a == 0).sum() > 1:
            return False
            
    return True

def _greedy_search(seed, deadline, start_set=None):
    """Greedy search with numpy validation."""
    rng = random.Random(seed)
    
    if start_set is None:
        # Find a small valid start set
        # Try random small sets
        start_set = []
        # Simple heuristic: start with random 2 or 3 elements
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        
        # Try to build a size 3 set quickly
        for i in range(3):
            for j in range(i+1, 30): # Look ahead a bit
                 # This is getting complicated. Let's just use a known valid set or random until valid.
                 pass
        
        # Fallback: Random restarts for small size
        best_small = []
        for _ in range(100):
            if time.time() > deadline: break
            cand = rng.sample(candidates, 3)
            if _is_valid_numpy(cand):
                best_small = cand
                break
        if not best_small:
             # Try size 2
             for _ in range(100):
                 if time.time() > deadline: break
                 cand = rng.sample(candidates, 2)
                 if _is_valid_numpy(cand):
                     best_small = cand
                     break
        if not best_small:
            return [4, 6, 9] # Guaranteed valid
            
        start_set = best_small

    current = list(start_set)
    
    # Pre-compute sums for current
    sums = {0}
    for x in current:
        sums = sums | {s + x for s in sums}
    sums.discard(0)
    ss_np = np.array(list(sums), dtype=np.int64)
    
    candidates = [x for x in range(1, 257) if x not in current]
    rng.shuffle(candidates)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add one element
        added = False
        for x in candidates:
            if time.time() > deadline: break
            
            # Check if adding x is valid
            # New sums = old sums + {s+x for s in old sums}
            # We need to check the union.
            # For speed, we can just construct the new set and check.
            # This is O(2^k) construction + O(2^k * 2^k / 64) check?
            
            new_sums = {s + x for s in sums}
            all_sums = sums | new_sums
            all_sums.discard(0)
            
            if len(all_sums) < 2:
                # Trivially valid?
                current.append(x)
                sums = all_sums
                ss_np = np.array(list(sums), dtype=np.int64)
                candidates.remove(x)
                improved = True
                added = True
                break
                
            # Check validity of all_sums
            # We can optimize by only checking new interactions, but full check is safer and numpy is fast
            ss_check = np.array(list(all_sums), dtype=np.int64)
            valid = True
            for i in range(len(ss_check)):
                a = ss_check[i]
                if (ss_check % a == 0).sum() > 1:
                    valid = False
                    break
                    
            if valid:
                current.append(x)
                sums = all_sums
                ss_np = np.array(list(sums), dtype=np.int64)
                candidates.remove(x)
                improved = True
                added = True
                break
        
        if not added:
            # Try a swap to unlock
            # Pick a random existing element to remove
            if len(current) > 1:
                idx = rng.randrange(len(current))
                u = current[idx]
                # Try to replace u with a random candidate
                for v in candidates:
                    if time.time() > deadline: break
                    new_current = current[:idx] + current[idx+1:] + [v]
                    if _is_valid_numpy(new_current):
                        current = new_current
                        # Recompute sums
                        sums = {0}
                        for x in current:
                            sums = sums | {s + x for s in sums}
                        sums.discard(0)
                        ss_np = np.array(list(sums), dtype=np.int64)
                        
                        # Update candidates
                        candidates.remove(v)
                        candidates.append(u)
                        improved = True
                        break
                        
    return current

def construct():
    start = time.time()
    deadline = start + 85.0
    
    best = [4, 6, 9]
    best_len = 3
    
    seed = 0
    while time.time() < deadline:
        seed += 1
        cur = _greedy_search(seed, deadline)
        if len(cur) > best_len:
            # Verify
            if _is_valid_numpy(cur):
                best = cur
                best_len = len(cur)
                
        # If we hit a high size, we can tighten the deadline for subsequent checks?
        # Or just keep going.
        
    return list(best)

# EVOLVE-BLOCK-END
