import random
import time

N = 256

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe against 60s kill
    
    # 1. Get baseline from standard greedy
    best_set = greedy_baseline()
    best_size = len(best_set)
    
    # 2. Randomized search for smaller maximal sets
    # We try to find a core and saturate it.
    
    while time.time() < deadline:
        # Generate a random core
        # Size of core: start small, e.g., 8-15
        core_size = random.randint(8, 15)
        core = generate_random_sidon_core(core_size)
        
        if not core:
            continue
            
        # Saturate the core
        # Strategy: Randomly pick a valid element to add
        current_set = set(core)
        current_sums = compute_sums(core)
        
        # Pool of candidates
        candidates = [x for x in range(1, N + 1) if x not in current_set]
        
        # We need to be careful: adding an element changes the valid candidates.
        # Instead of maintaining a static list, we check validity on the fly.
        # To speed up, we can shuffle candidates and iterate.
        
        # However, "random valid" is tricky if we just iterate a shuffled list once.
        # If we skip an element because it's invalid now, it might become valid later?
        # No, adding elements only *adds* sums, so if x is invalid now (conflicts with existing sums),
        # it will remain invalid later (conflicts with existing + new sums).
        # So, if x is invalid, it is permanently blocked.
        # Thus, we can just iterate through a shuffled list of candidates,
        # and add x if it is currently valid.
        
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 1.0:
                break
            
            # Check if x can be added to current_set
            if can_add(x, current_set, current_sums):
                current_set.add(x)
                # Update sums
                # New sums: 2*x and x+s for s in current_set (excluding x itself? No, x is new)
                # The set current_set already contains the old elements.
                # We need to add sums involving x and the OLD elements.
                for s in current_set:
                    if s != x:
                        current_sums.add(x + s)
                current_sums.add(2 * x)
                
        # After this loop, current_set is maximal.
        # Check size
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
            # If we hit the theoretical lower bound, stop
            if best_size <= 16: 
                break

    return best_set

def greedy_baseline():
    selected = []
    sums = set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def compute_sums(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    return sums

def can_add(x, S, sums):
    # Check 2x
    if 2 * x in sums:
        return False
    # Check x + s
    for s in S:
        if x + s in sums:
            return False
    return True

def generate_random_sidon_core(target_size):
    core = []
    sums = set()
    # Try to pick random elements
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if len(core) == target_size:
            break
        # Check if x can be added to core
        if can_add(x, core, sums):
            core.append(x)
            # Update sums
            for s in core:
                if s != x:
                    sums.add(x + s)
            sums.add(2 * x)
            
    return core if len(core) == target_size else []
