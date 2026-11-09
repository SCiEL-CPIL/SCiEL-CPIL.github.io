import random
import time
import sys

N = 512

# Precompute all 4-term arithmetic progressions
APs = []
aps_containing = [[] for _ in range(N + 1)]

for d in range(1, (N - 1) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        ap = (a, a + d, a + 2 * d, a + 3 * d)
        idx = len(APs)
        APs.append(ap)
        for x in ap:
            aps_containing[x].append(idx)

def is_valid_set(S_set):
    """Check if S_set contains any 4-AP."""
    for ap in APs:
        if all(x in S_set for x in ap):
            return False
    return True

def can_add(S_set, x):
    """Check if adding x to S_set keeps it 4-AP free."""
    for idx in aps_containing[x]:
        ap = APs[idx]
        count = 0
        for y in ap:
            if y != x and y in S_set:
                count += 1
        if count == 3:
            return False
    return True

def find_conflicts(S_set, x):
    """Return list of AP indices that would be completed if x is added."""
    conflicts = []
    for idx in aps_containing[x]:
        ap = APs[idx]
        count = 0
        for y in ap:
            if y != x and y in S_set:
                count += 1
        if count == 3:
            conflicts.append(idx)
    return conflicts

def check_swap_validity(S_set, x, y):
    """
    Check if (S_set - {y}) U {x} is valid.
    S_set is known to be valid.
    We only need to check APs containing x.
    """
    # Optimization: Instead of creating a new set, just check conditions
    # For an AP containing x to be fully in new set:
    # 1. x is in AP (true by definition)
    # 2. Other 3 elements are in S_set \ {y}
    
    for idx in aps_containing[x]:
        ap = APs[idx]
        # Check if other 3 are in S_set and none is y
        others_in_S = 0
        y_in_ap = False
        for z in ap:
            if z == x:
                continue
            if z in S_set:
                if z == y:
                    y_in_ap = True
                else:
                    others_in_S += 1
        # We need 3 others in S_set \ {y}.
        # If y is in ap, it is removed. So we need 3 others that are NOT y.
        # If y is not in ap, we need 3 others that ARE in S_set.
        
        # Let's re-evaluate:
        # New set S' = S \ {y} U {x}.
        # AP is valid in S' if all 4 elements are in S'.
        # Elements are x, z1, z2, z3.
        # x is in S'.
        # z_i must be in S'.
        # z_i in S' iff (z_i in S and z_i != y) or z_i == x.
        # Since z_i != x, condition is z_i in S and z_i != y.
        
        count_valid_others = 0
        for z in ap:
            if z == x:
                continue
            if z in S_set and z != y:
                count_valid_others += 1
        
        if count_valid_others == 3:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin

    best_size = 0
    best_set = set()

    # Run multiple iterations of ILS
    while time.time() < deadline:
        # 1. Initialization: Greedy with random shuffle
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        # Optional: Bias towards Cantor set?
        # Let's try standard random greedy first, it's robust.
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

        # 2. Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add elements
            # We can try all complement elements or a random subset
            # To be fast, try a random subset of complement
            complement = [x for x in range(1, N + 1) if x not in S]
            if complement:
                random.shuffle(complement)
                # Try first K candidates
                for x in complement[:50]: 
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        break
                # If none of the first 50 worked, maybe try more? 
                # Or just break and try swaps.
            
            if not improved:
                # Try swaps
                # Pick a random x from complement
                if complement:
                    x = random.choice(complement)
                    conflicts = find_conflicts(S, x)
                    if conflicts:
                        # Try to resolve by swapping one of the conflicting elements
                        # For each conflict AP, we must remove one of the 3 elements in S.
                        # Let's pick a random conflict
                        ap_idx = random.choice(conflicts)
                        ap = APs[ap_idx]
                        # Elements in S that are part of this AP (excluding x)
                        removable = [z for z in ap if z != x and z in S]
                        if removable:
                            # Try removing each one? Or just one?
                            # Let's try all removable elements in this AP
                            for y in removable:
                                if check_swap_validity(S, x, y):
                                    S.remove(y)
                                    S.add(x)
                                    improved = True
                                    break
                    # If no swap found, we are stuck in local optimum for this structure
                    # We can try a "kick" or just break to restart
                    if not improved:
                        break

        # 3. Perturbation (for ILS)
        # Remove a few random elements to allow new structure
        if len(S) > 10 and time.time() < deadline:
            to_remove = random.sample(list(S), k=min(5, len(S)//10))
            for y in to_remove:
                S.remove(y)
            # Re-run local search on perturbed set? 
            # The loop above will run again if we structure it as a while loop.
            # But the current structure is: Init -> Local Search -> Update Best.
            # Let's add an inner loop for local search that includes perturbation if stuck.
            
    return sorted(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    S_set = set(res)
    valid = True
    for ap in APs:
        if all(x in S_set for x in ap):
            valid = False
            break
    print(f"Valid: {valid}")
