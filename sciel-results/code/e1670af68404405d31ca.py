"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 512

def _can_add(S, x):
    """
    Check if x can be added to S without creating a 4-AP.
    S is a set of integers.
    """
    # Case 1: x is the 1st term: x, x+d, x+2d, x+3d
    # Need x+d, x+2d, x+3d in S
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
            return False
    
    # Case 2: x is the 2nd term: x-d, x, x+d, x+2d
    # Need x-d, x+d, x+2d in S
    # x-d >= 1 => d <= x-1
    # x+2d <= N => d <= (N-x)//2
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if (x - d in S) and (x + d in S) and (x + 2*d in S):
            return False
    
    # Case 3: x is the 3rd term: x-2d, x-d, x, x+d
    # Need x-2d, x-d, x+d in S
    # x-2d >= 1 => d <= (x-1)//2
    # x+d <= N => d <= N-x
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        if (x - 2*d in S) and (x - d in S) and (x + d in S):
            return False
    
    # Case 4: x is the 4th term: x-3d, x-2d, x-d, x
    # Need x-3d, x-2d, x-d in S
    # x-3d >= 1 => d <= (x-1)//3
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
            return False
            
    return True

def _greedy_construct():
    S = set()
    for x in range(1, N + 1):
        if _can_add(S, x):
            S.add(x)
    return S

def _find_blockers(S, x):
    """
    Find elements in S that block x from being added.
    Returns a list of blockers.
    """
    blockers = set()
    
    # Case 1: x is 1st. Blockers are in {x+d, x+2d, x+3d}
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        b1 = x + d
        b2 = x + 2*d
        b3 = x + 3*d
        if b1 in S and b2 in S and b3 in S:
            blockers.update([b1, b2, b3])
            
    # Case 2: x is 2nd. Blockers are in {x-d, x+d, x+2d}
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        b1 = x - d
        b2 = x + d
        b3 = x + 2*d
        if b1 in S and b2 in S and b3 in S:
            blockers.update([b1, b2, b3])
            
    # Case 3: x is 3rd. Blockers are in {x-2d, x-d, x+d}
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        b1 = x - 2*d
        b2 = x - d
        b3 = x + d
        if b1 in S and b2 in S and b3 in S:
            blockers.update([b1, b2, b3])
            
    # Case 4: x is 4th. Blockers are in {x-3d, x-2d, x-d}
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        b1 = x - 3*d
        b2 = x - 2*d
        b3 = x - d
        if b1 in S and b2 in S and b3 in S:
            blockers.update([b1, b2, b3])
            
    return list(blockers)

def _local_search(S, deadline):
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any element
        unselected = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(unselected)
        
        for x in unselected:
            if _can_add(S, x):
                S.add(x)
                improved = True
                break
        
        if improved:
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            continue
        
        # Phase 2: Try directed swaps based on blockers
        # For each unselected x, find blockers. If we remove a blocker, can we add x?
        swap_done = False
        for x in unselected:
            blockers = _find_blockers(S, x)
            if not blockers:
                # Should have been caught in Phase 1, but safety check
                S.add(x)
                improved = True
                break
            
            # Try removing each blocker
            random.shuffle(blockers)
            for b in blockers:
                # Try swap: remove b, add x
                S.remove(b)
                if _can_add(S, x):
                    S.add(x)
                    # Size is same, but we made a change
                    swap_done = True
                    break
                else:
                    # Revert
                    S.add(b)
            
            if swap_done:
                break
        
        if swap_done:
            # Check if this swap allows future growth, but for now we just accept the state
            # We don't update best_S here because size didn't increase
            # But we continue the loop to see if we can add more
            
            # Occasionally, if we are stuck for a while, we might need a kick
            # For now, just continue
            continue
        
        # Phase 3: No additions, no simple swaps found.
        # Try a random kick: remove a random element, try to add a random element
        # This might reduce size temporarily but escape local minima
        if random.random() < 0.1: # 10% chance to kick
            if S:
                rem = random.choice(list(S))
                S.remove(rem)
                # Try to add something
                unselected = [x for x in range(1, N + 1) if x not in S]
                random.shuffle(unselected)
                for x in unselected:
                    if _can_add(S, x):
                        S.add(x)
                        break

    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Initial greedy construction
    S = _greedy_construct()
    
    # Run local search
    best_S = _local_search(S, deadline)
    
    # If we have time, try another restart with a different random seed or perturbation
    if time.time() < deadline - 5.0:
        # Perturb the best set found
        S2 = set(best_S)
        # Remove a few random elements
        for _ in range(3):
            if S2:
                S2.remove(random.choice(list(S2)))
        
        # Run local search again
        best_S2 = _local_search(S2, deadline)
        
        if len(best_S2) > len(best_S):
            best_S = best_S2

    return sorted(best_S)
# EVOLVE-BLOCK-END
