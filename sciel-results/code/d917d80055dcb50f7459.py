import time
import random

N = 256

def _is_sidon(selected):
    """Check if selected is a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        if 2 * a in sums:
            return False
        sums.add(2 * a)
        for j in range(i + 1, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _compute_sums(selected):
    sums = set()
    for i in range(len(selected)):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, len(selected)):
            sums.add(a + selected[j])
    return sums

def _greedy_maximal_sidon(seed_val):
    """
    Construct a maximal Sidon set using a greedy set-cover approach.
    We want to cover all elements in [1, N] with "blocks".
    An element y is blocked by set S if S U {y} is not Sidon.
    """
    rng = random.Random(seed_val)
    
    # Current Sidon set
    S = []
    # Current sums of S
    sums = set()
    # Elements that are NOT yet blocked (i.e., can be added to S without violating Sidon? No.)
    # Definition: y is unblocked if S U {y} IS Sidon.
    # We want S to be maximal, meaning NO y in [1, N] \ S is unblocked.
    # So we want to cover all y in [1, N] such that they become "blocked".
    # Initially S is empty. All y in [1, N] are unblocked (can be added).
    # We pick x to add to S. This action blocks some other y's.
    
    # unblocked[y] = True if S U {y} is Sidon.
    # Note: If y is in S, it's "used".
    # We only care about y not in S.
    
    unblocked = [True] * (N + 1)
    # 0 is unused
    for i in range(1, N + 1):
        unblocked[i] = True
        
    # We need to ensure S itself is Sidon.
    # When we add x, we must ensure x is unblocked wrt current S.
    # i.e., S U {x} is Sidon.
    
    while any(unblocked[1:]):
        best_x = -1
        best_blocked_count = -1
        candidates = []
        
        for x in range(1, N + 1):
            if x in S:
                continue
            if not unblocked[x]:
                continue
            
            # Check if S U {x} is Sidon
            # New sums: 2x, x+s for s in S
            new_sums = {2 * x}
            valid = True
            for s in S:
                val = x + s
                if val in new_sums or val in sums:
                    valid = False
                    break
                new_sums.add(val)
            
            if not valid:
                continue
                
            # If valid, how many unblocked elements does this x block?
            # An element y (y != x, y not in S) is blocked if (S U {x}) U {y} is NOT Sidon.
            # This is equivalent to checking if any sum in (Sums(S) U NewSums(x)) 
            # equals any sum formed by adding y to (S U {x}).
            
            # Sums of S' = S U {x} is sums | new_sums
            # Sums formed by adding y to S' are: 2y, y+s' for s' in S'
            
            # We only count y if unblocked[y] is currently True.
            # Note: y=x is already handled (x is being added).
            # We need to check y in [1, N] \ S.
            
            blocked_count = 0
            # Optimization: We can just check all y. N=256 is small.
            for y in range(1, N + 1):
                if y in S or y == x:
                    continue
                if not unblocked[y]:
                    continue
                
                # Check if y is blocked by S'
                # Conflicts if:
                # 1. 2y is in sums | new_sums
                # 2. y+s' is in sums | new_sums for any s' in S'
                
                # 1. 2y
                if 2 * y in sums or 2 * y in new_sums:
                    blocked_count += 1
                    continue # Already blocked, no need to check further for this y
                
                # 2. y + s'
                # s' can be x or s in S
                # y + x
                if (y + x) in sums or (y + x) in new_sums:
                    blocked_count += 1
                    continue
                    
                # y + s for s in S
                is_blocked = False
                for s in S:
                    if (y + s) in sums or (y + s) in new_sums:
                        is_blocked = True
                        break
                if is_blocked:
                    blocked_count += 1
            
            if blocked_count > best_blocked_count:
                best_blocked_count = blocked_count
                best_x = x
                candidates = [x]
            elif blocked_count == best_blocked_count:
                candidates.append(x)
                
        if best_x == -1:
            # This shouldn't happen if unblocked is not empty, because 
            # at least one unblocked element can be added to a Sidon set?
            # Actually, if S is maximal, unblocked should be empty.
            # If we are stuck, it means no valid x exists? 
            # But S is Sidon, so we can always add elements until we can't.
            # If we can't add any element, S is maximal. 
            # But if unblocked is not empty, it means some y can be added.
            # So best_x should not be -1.
            break
            
        # Random tie-breaking
        x_to_add = rng.choice(candidates)
        
        # Add x to S
        S.append(x_to_add)
        
        # Update sums
        # new_sums calculated in the loop, but we need to re-calc or store?
        # We can just re-calc sums for S. It's O(|S|^2).
        # Or update incrementally.
        # Incremental:
        sums.add(2 * x_to_add)
        for s in S[:-1]: # S[:-1] is S before adding x_to_add
            sums.add(x_to_add + s)
            
        # Update unblocked
        # We need to determine which y's are NOW blocked.
        # A y is blocked if (S U {y}) is not Sidon.
        # We can re-calculate unblocked for all y not in S.
        # This is O(N * |S|).
        
        for y in range(1, N + 1):
            if y in S:
                continue
            # Check if S U {y} is Sidon
            # Sums of S are in 'sums'.
            # New sums from y: 2y, y+s for s in S.
            
            # If any of these are in 'sums', it's not Sidon -> Blocked.
            # Also, if 2y == y+s (i.e., y=s, impossible as y not in S) or 2y == 2y (trivial).
            # We just check intersection with 'sums'.
            
            is_sidon_with_y = True
            if 2 * y in sums:
                is_sidon_with_y = False
            else:
                for s in S:
                    if (y + s) in sums:
                        is_sidon_with_y = False
                        break
            
            unblocked[y] = is_sidon_with_y

    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = None
    best_size = N + 1
    
    # Run multiple trials
    # Estimate time per trial:
    # Steps ~ |S|. |S| ~ 15-20.
    # Inner loop: N candidates * (Sidon check + Blocked count).
    # Sidon check: O(|S|). Blocked count: O(N * |S|) roughly.
    # So one step is O(N * |S| * |S|) ? No, blocked count is O(N * |S|).
    # Total per step: O(N * |S|).
    # Total per trial: O(|S|^2 * N).
    # 15^2 * 256 ~ 57k ops. Very fast.
    # We can run many trials.
    
    trial = 0
    while time.time() < deadline:
        # Increase seed to get different random choices
        S = _greedy_maximal_sidon(trial)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            # Optional: If we find a very small set, we might want to try to reduce it further?
            # Greedy set cover is usually near-optimal for this kind of problem.
            
        trial += 1
        
        # If we've found a set of size 10 or less, we can be more aggressive with time or stop?
        # Minimal maximal Sidon sets in [1, N] are known to be small.
        # For N=256, the minimum size is likely around 10-15.
        
    # Final verification
    if best_set is not None:
        # Ensure it's Sidon
        if not _is_sidon(best_set):
            return [] # Should not happen
        # Ensure it's maximal
        # The algorithm ensures maximality by construction (stops when unblocked is empty).
        # But let's double check to be safe.
        S_set = set(best_set)
        sums = _compute_sums(best_set)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x can be added
            if 2 * x in sums:
                continue
            conflict = False
            for s in best_set:
                if x + s in sums:
                    conflict = True
                    break
            if not conflict:
                # Not maximal
                return []
                
    return best_set

