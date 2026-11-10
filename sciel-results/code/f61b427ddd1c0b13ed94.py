import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0 # Safety margin
    
    n = 512
    # Precompute all 4-APs
    # An AP is (a, d) -> a, a+d, a+2d, a+3d
    # We store for each number, the list of APs it belongs to.
    # AP identified by (a, d).
    # To check validity of S, we just need to know if any AP is fully in S.
    
    # For efficiency in local search:
    # For each x, store list of APs containing x.
    # When adding x, we need to check all APs in list[x] to see if the other 3 are in S.
    
    aps_for_num = [[] for _ in range(n + 1)]
    all_aps = []
    
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            all_aps.append(ap)
            for x in ap:
                aps_for_num[x].append(ap)
                
    # Initial set: Base-4 {0,1} digits
    S = set()
    for i in range(32):
        # i in binary gives bits b4..b0
        val = 0
        for j in range(5):
            if (i >> j) & 1:
                val += (1 << (2 * j)) # 4^j
        # val is in [0, 341]
        S.add(val + 1)
        
    def is_valid_addition(x, S):
        if x in S:
            return False
        for ap in aps_for_num[x]:
            # Check if other 3 are in S
            # ap is a tuple of 4. x is one of them.
            # We can just check if all 4 are in S after adding x?
            # No, x is not in S yet.
            # So we check if the other 3 are in S.
            if all(y in S for y in ap if y != x):
                return False
        return True

    def try_greedy_add(S):
        added = False
        # Try to add any valid number
        # Randomize order to escape symmetry? Or just 1..n?
        # Let's try 1..n first
        for x in range(1, n + 1):
            if x not in S and is_valid_addition(x, S):
                S.add(x)
                added = True
        return added

    # Phase 1: Greedy add
    while try_greedy_add(S):
        pass
        
    # Phase 2: Local Search (Swaps)
    # Try to remove one, add more
    improved = True
    while improved:
        improved = False
        for y in list(S):
            if time.time() > timeout:
                break
            S.remove(y)
            # Try to add back y or others
            # Actually, just run greedy add on the remaining set
            # But we want to see if we can get a larger set
            # Save current size
            size_before = len(S)
            
            # Try to add any valid numbers
            added_any = False
            for x in range(1, n + 1):
                if x not in S and is_valid_addition(x, S):
                    S.add(x)
                    added_any = True
                    if time.time() > timeout:
                        break
            
            if len(S) > size_before:
                improved = True
                # We found a better set, keep it
            else:
                # Revert? 
                # The greedy add might have added y back or others.
                # If size didn't increase, we should probably revert to previous state
                # to avoid getting stuck in a worse local minimum?
                # Actually, if we removed y and added nothing, size decreased.
                # If we added y back, size is same.
                # If we added x != y, size is same.
                # We want to increase size.
                # So if not improved, revert.
                # But reverting is hard if we mutated S.
                # Let's snapshot S before removing y?
                # Copying a set of ~50 elements is cheap.
                pass
        
        if time.time() > timeout:
            break
            
    # The swap logic above is flawed because it modifies S in place and doesn't revert properly if no improvement.
    # Let's rewrite the local search properly.
    
    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for y in list(S):
                if time.time() > timeout:
                    return S
                S_copy = set(S)
                S_copy.remove(y)
                # Greedy fill
                for x in range(1, n + 1):
                    if x not in S_copy and is_valid_addition(x, S_copy):
                        S_copy.add(x)
                if len(S_copy) > len(S):
                    S = S_copy
                    improved = True
                    break # Restart loop with new S
        return S

    S = local_search(S)
    
    # Phase 3: Randomized restarts / Kicks
    best_S = S
    while time.time() < timeout:
        # Kick: remove 3 random elements
        S_kick = set(best_S)
        if len(S_kick) < 3:
            break
        removals = random.sample(list(S_kick), 3)
        for r in removals:
            S_kick.remove(r)
        
        # Greedy fill
        for x in range(1, n + 1):
            if x not in S_kick and is_valid_addition(x, S_kick):
                S_kick.add(x)
                
        # Local search
        S_kick = local_search(S_kick)
        
        if len(S_kick) > len(best_S):
            best_S = S_kick
            
    return sorted(best_S)
