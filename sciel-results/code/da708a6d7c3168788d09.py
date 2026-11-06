"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    def get_sums(selected):
        sums = set()
        for i in range(len(selected)):
            a = selected[i]
            sums.add(2 * a)
            for j in range(i + 1, len(selected)):
                sums.add(a + selected[j])
        return sums

    def is_sidon(selected):
        if not selected:
            return True
        sums = get_sums(selected)
        expected = len(selected) * (len(selected) + 1) // 2
        return len(sums) == expected

    def get_extendable(selected, sums):
        extendable = []
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            new_sums = {2 * x}
            new_sums.update(x + y for y in selected)
            
            if new_sums & sums:
                continue
            
            extendable.append(x)
        return extendable

    def greedy_max_blocking():
        S = []
        S_set = set()
        Sums = set()
        Diff_S = set()
        alive = set(range(1, N + 1))
        
        while alive:
            best_x = None
            max_blocked = -1
            best_blocked_set = set()
            
            # To speed up, we can limit the candidates if alive is large, but N=256 is small.
            # We iterate over all alive to find the best.
            # Randomize order to break ties and explore different paths
            candidates = list(alive)
            random.shuffle(candidates)
            
            for x in candidates:
                # Check if x is still valid against S
                # New sums if x added: 2x, x+s for s in S
                new_sums_x = {2 * x}
                new_sums_x.update(x + s for s in S)
                
                if new_sums_x & Sums:
                    # x is invalid, should have been removed. Skip.
                    continue
                
                # Calculate which y in alive (y != x) are blocked by x
                blocked = set()
                for y in candidates:
                    if y == x:
                        continue
                    
                    # y is blocked if it cannot be added to S ∪ {x}
                    # Conditions derived:
                    # 1. 2y collides with new sums: 2y = x+s => s = 2y-x. Check if 2y-x in S_set.
                    # 2. y+z collides with new sums:
                    #    a) y+z = 2x => z = 2x-y. Check if 2x-y in S_set. (Note: z=x implies y=x, skipped)
                    #    b) y+z = x+s => s-z = y-x. Check if (y-x) in Diff_S.
                    
                    if (2 * y - x) in S_set:
                        blocked.add(y)
                        continue
                        
                    if (2 * x - y) in S_set:
                        blocked.add(y)
                        continue
                        
                    if (y - x) in Diff_S:
                        blocked.add(y)
                        continue

                if len(blocked) > max_blocked:
                    max_blocked = len(blocked)
                    best_x = x
                    best_blocked_set = blocked
            
            if best_x is None:
                # Fallback: pick any valid x
                # Find first valid x
                for x in candidates:
                    new_sums_x = {2 * x}
                    new_sums_x.update(x + s for s in S)
                    if not (new_sums_x & Sums):
                        best_x = x
                        break
                if best_x is None:
                    # No valid x? This means S is maximal but alive is not empty?
                    # This implies a bug in alive maintenance.
                    # Break to avoid infinite loop.
                    break

            # Add best_x to S
            S.append(best_x)
            S_set.add(best_x)
            
            # Update Sums
            new_sums_best = {2 * best_x}
            new_sums_best.update(best_x + s for s in S[:-1])
            Sums.update(new_sums_best)
            
            # Update Diff_S
            # New differences involving best_x:
            # s - best_x and best_x - s for all s in S[:-1]
            # Also 0 is not needed for blocking distinct y!=x, but let's keep it clean.
            for s in S[:-1]:
                Diff_S.add(s - best_x)
                Diff_S.add(best_x - s)
            
            # Update alive
            alive.remove(best_x)
            for y in best_blocked_set:
                if y in alive:
                    alive.remove(y)

        return S

    best_set = None
    best_size = N + 1
    
    # Run the greedy algorithm multiple times with different random seeds/orderings
    # Since the inner loop is now O(1) per pair, we can afford many restarts.
    
    # Estimate time per restart:
    # Initial size ~ N. Steps ~ N/2? No, size of S is ~ sqrt(N) ~ 16.
    # Number of steps is ~16.
    # In each step, we iterate over alive (size ~ N) and for each x, iterate over alive (size ~ N) to compute blocked.
    # So ~ N^2 operations per step. 16 * 256^2 ~ 1M ops.
    # Python can do ~10M simple ops/sec. So ~0.1s per restart.
    # We have ~55s. So we can do ~500 restarts.
    
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        S = greedy_max_blocking()
        
        if S:
            # Verify
            if is_sidon(S):
                sums = get_sums(S)
                if not get_extendable(S, sums):
                    if len(S) < best_size:
                        best_size = len(S)
                        best_set = S
                        # If we find a very small set, we might want to stop early?
                        # Minimal maximal Sidon sets in [1,256] are likely around 15-20.
                        # If we hit 15, it's hard to beat.
                        if best_size <= 14:
                            break

    if best_set is None:
        # Fallback: Simple greedy Sidon (maximal by construction if we remove blocked)
        S = []
        Sums = set()
        alive = set(range(1, N + 1))
        while alive:
            # Pick smallest available
            x = min(alive)
            new_sums_x = {2 * x}
            new_sums_x.update(x + s for s in S)
            if new_sums_x & Sums:
                alive.remove(x)
                continue
            
            S.append(x)
            Sums.update(new_sums_x)
            alive.remove(x)
            
            # Remove blocked
            to_remove = []
            for y in alive:
                if 2 * y in Sums:
                    to_remove.append(y)
                    continue
                blocked = False
                for s in S:
                    if (y + s) in Sums:
                        blocked = True
                        break
                if blocked:
                    to_remove.append(y)
            for y in to_remove:
                alive.remove(y)
        
        best_set = S

    return best_set

# EVOLVE-BLOCK-END
