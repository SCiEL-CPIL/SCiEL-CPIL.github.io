import time
import random
import sys

# Increase recursion limit just in case, though not strictly needed for this logic
sys.setrecursionlimit(2000)

class Verifier:
    __slots__ = ("n", "dp", "acc", "valid")

    def __init__(self, n):
        self.n = n
        self.dp = [0] * (n + 1)
        self.dp[0] = 1
        self.acc = 1
        self.valid = True

    def add(self, x):
        if not self.valid:
            return False
        n = self.n
        prev = self.dp
        ndp = list(prev)
        for k in range(n, 0, -1):
            ndp[k] = prev[k] | (prev[k - 1] << x)
        
        acc = self.acc
        for k in range(n + 1):
            if ndp[k] & acc:
                self.valid = False
                return False
            acc |= ndp[k]
        self.dp = ndp
        self.acc = acc
        return True

    def is_valid_state(self):
        return self.valid

def _check(arr):
    """Independent full verification."""
    n = len(arr)
    if n < 1 or n > 60:
        return False
    seen = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in seen:
            return False
        seen.add(v)
    
    # Re-verify DP
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _build_verifier(arr):
    """Build a verifier from a list of ints. Returns (ver, valid)."""
    ver = Verifier(60)
    for x in arr:
        if not ver.add(x):
            return ver, False
    return ver, True

def _greedy_grow(cur_set, used, ver, candidates, deadline):
    """
    Tries to add any element from candidates to cur_set.
    Returns (new_cur_set, new_used, new_ver, added_count)
    """
    added_count = 0
    # We sort candidates to try "easy" ones first? Or just iterate?
    # Iterating through a list of unused elements.
    # To be efficient, we can just check all unused elements repeatedly.
    # But checking 256 elements x 60 steps x bit ops is fast.
    
    # Optimization: Keep a list of unused candidates.
    # If we add one, remove it from the list.
    
    # Since we don't want to modify the original 'candidates' list structure too much,
    # we will just loop through range(1, 257) and check if in used.
    # This is O(256) per check.
    
    # Better: Maintain a list of unused candidates.
    # But 'used' is a set.
    
    # Let's just iterate through all possible values 1..256.
    # If time is tight, we can break.
    
    changed = True
    while changed and time.monotonic() < deadline:
        changed = False
        for v in range(256, 0, -1): # Try larger first? Or smaller? 
            # The order of greedy growth can matter. 
            # Let's try a mix. For now, descending.
            if v in used:
                continue
            
            # Try to add v
            # We need to check if ver.add(v) would succeed.
            # We can do this by cloning the verifier state? 
            # Or just try and revert?
            # Reverting bitsets is hard.
            # Cloning: dp is list of ints. Shallow copy is fine.
            
            old_dp = ver.dp
            old_acc = ver.acc
            old_valid = ver.valid
            
            # Try add
            # We need to be careful: ver.add modifies ver.
            # If it fails, we must restore.
            
            # Actually, let's just use a helper that checks validity without modifying?
            # No, Verifier.add is the only way.
            # Let's save state.
            
            # Save state
            dp_copy = ver.dp[:]
            acc_copy = ver.acc
            valid_copy = ver.valid
            
            if ver.add(v):
                # Success
                cur_set.append(v)
                used.add(v)
                added_count += 1
                changed = True
                # Don't break, keep trying to add more
            else:
                # Restore state
                ver.dp = dp_copy
                ver.acc = acc_copy
                ver.valid = valid_copy
    
    return cur_set, used, ver, added_count

def _local_search_ils(start_set, deadline):
    """
    Iterated Local Search.
    1. Greedy Grow.
    2. If stuck, Swap (Remove 1, Add 1). If valid, accept and go to 1.
    """
    cur_set = list(start_set)
    used = set(cur_set)
    ver, valid = _build_verifier(cur_set)
    if not valid:
        return cur_set
    
    best_set = list(cur_set)
    best_len = len(cur_set)
    
    # Main ILS Loop
    # We limit the number of swaps per "stuck" phase to avoid infinite loops on bad configs
    max_swaps_per_stuck = 1000
    
    while time.monotonic() < deadline:
        # 1. Greedy Grow
        cur_set, used, ver, added = _greedy_grow(cur_set, used, ver, None, deadline)
        
        if added > 0:
            if len(cur_set) > best_len:
                best_len = len(cur_set)
                best_set = list(cur_set)
            # If we added, we might be able to add more, so loop back to grow
            continue
            
        # 2. Stuck. Try Swaps.
        # We try to find a swap that results in a valid set.
        # Once found, we accept it and go back to Grow.
        
        swaps_tried = 0
        found_swap = False
        
        # Shuffle indices for randomization
        indices = list(range(len(cur_set)))
        random.shuffle(indices)
        
        # Candidates for swap-in: all unused numbers
        unused_candidates = [v for v in range(1, 257) if v not in used]
        if not unused_candidates:
            break # No more elements to add anyway
        
        random.shuffle(unused_candidates)
        
        for i_idx in indices:
            if found_swap:
                break
            a_i = cur_set[i_idx]
            
            for v in unused_candidates:
                if time.monotonic() > deadline:
                    return best_set
                
                # Try swap: remove a_i, add v
                # New set: cur_set without a_i, plus v
                # Check if this new set is valid.
                
                # Optimization: If v == a_i, skip (not possible as v is unused)
                
                # Build new set list for verification
                # To avoid list copy overhead, we can just build the verifier directly
                new_ver = Verifier(60)
                valid_new = True
                for x in cur_set:
                    if x == a_i:
                        continue
                    if not new_ver.add(x):
                        valid_new = False
                        break
                if valid_new:
                    if not new_ver.add(v):
                        valid_new = False
                
                if valid_new:
                    # Swap is valid
                    # Update state
                    cur_set[i_idx] = v
                    used.discard(a_i)
                    used.add(v)
                    ver = new_ver
                    found_swap = True
                    break # Accept this swap and go back to Grow phase
            
            swaps_tried += 1
            if swaps_tried >= max_swaps_per_stuck:
                break
        
        if not found_swap:
            # Couldn't find a valid swap in this batch.
            # Break out of ILS loop for this start_set? 
            # Or try a more aggressive perturbation?
            # For now, break to try a different start_set or restart.
            break
            
    return best_set

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0 # 5s safety margin
    
    best_global = []
    best_len = 0
    
    def update_best(arr):
        nonlocal best_global, best_len
        if len(arr) > best_len:
            if _check(arr): # Final safety check
                best_global = list(arr)
                best_len = len(arr)
    
    # 1. Generate initial seeds
    seeds = []
    
    # Seed 1: Greedy Descending
    s1 = []
    u1 = set()
    v1 = Verifier(60)
    for v in range(256, 0, -1):
        if v1.add(v):
            s1.append(v)
            u1.add(v)
    seeds.append(s1)
    
    # Seed 2: Greedy Ascending
    s2 = []
    u2 = set()
    v2 = Verifier(60)
    for v in range(1, 257):
        if v2.add(v):
            s2.append(v)
            u2.add(v)
    seeds.append(s2)
    
    # Seed 3: Random Greedy (Fixed Seed for reproducibility in this run, but random in general)
    rng = random.Random(42)
    s3 = []
    u3 = set()
    v3 = Verifier(60)
    order = list(range(1, 257))
    rng.shuffle(order)
    for v in order:
        if v3.add(v):
            s3.append(v)
            u3.add(v)
    seeds.append(s3)
    
    # Seed 4: "Gap" Greedy - try to space them out
    # Start with 1, then 1+gap, etc.
    # Actually, just use the best of the above as a start for ILS.
    
    # Run ILS on the best seed
    # Sort seeds by length
    seeds.sort(key=len, reverse=True)
    
    best_seed = seeds[0]
    update_best(best_seed)
    
    # Run ILS starting from the best seed
    if time.monotonic() < deadline:
        result = _local_search_ils(best_seed, deadline)
        update_best(result)
        
    # If we have time, try ILS from other seeds
    if time.monotonic() < deadline and len(seeds) > 1:
        for s in seeds[1:]:
            if time.monotonic() > deadline - 5:
                break
            if len(s) >= best_len:
                continue # No point if already worse
            result = _local_search_ils(s, deadline)
            update_best(result)
            
    # If we have time, do random restarts of the whole process?
    # The ILS is quite strong. Let's add a few random restarts of the greedy seed generation.
    
    rng2 = random.Random(12345)
    while time.monotonic() < deadline - 10:
        # Generate a random greedy seed
        s_rand = []
        u_rand = set()
        v_rand = Verifier(60)
        order = list(range(1, 257))
        rng2.shuffle(order)
        for v in order:
            if v_rand.add(v):
                s_rand.append(v)
                u_rand.add(v)
        
        if len(s_rand) > best_len:
            update_best(s_rand)
            
        # Run ILS on this random seed
        if time.monotonic() < deadline - 5:
            result = _local_search_ils(s_rand, deadline)
            update_best(result)
            
        if best_len == 60:
            break

    if not best_global:
        # Fallback
        best_global = list(range(1, 31))
        
    return [int(v) for v in best_global]

# Entry point
if __name__ == "__main__":
    # Self-test
    res = construct()
    print(f"Result size: {len(res)}")
    if _check(res):
        print("Valid")
    else:
        print("INVALID")
