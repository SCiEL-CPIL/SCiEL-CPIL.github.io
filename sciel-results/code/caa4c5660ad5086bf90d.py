import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_valid_solution(arr):
    """
    Verifies:
    1. All elements are distinct integers in [1, N].
    2. It is a Sidon set (all unordered pair sums a+b with a<=b are distinct).
    3. It is maximal (adding any missing element from [1, N] destroys the Sidon property).
    """
    if not arr:
        return False
    
    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False
    
    for x in arr:
        if x < 1 or x > N:
            return False
            
    # Compute all sums
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
            
    # Check maximality
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        
        # Try to add x
        # New sums are {x+a for a in arr} U {2x}
        # If any of these are in `sums`, or if there's a conflict among them, x cannot be added.
        # If x CAN be added (no conflicts), the set is NOT maximal.
        # We want to ensure that for EVERY x not in arr, x CANNOT be added.
        
        # Check conflicts with existing sums
        conflict = False
        for a in arr:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
        
        if (2 * x) in sums:
            continue
            
        # Check conflicts among the new sums themselves
        # The new sums are S_new = {x+a | a in arr} U {2x}
        # Note that x+a == x+b iff a==b (since arr has distinct elements).
        # So conflicts can only happen between:
        # 1. x+a and 2x  => x+a = 2x => a = x (impossible since x not in arr)
        # 2. x+a and x+b => a=b (impossible)
        # So the new sums are always distinct from each other.
        # Thus, if no conflict with existing sums, x CAN be added.
        
        # Since x can be added, the set is not maximal.
        return False
        
    return True

def _greedy_construct(order, rng=None):
    """
    Greedily constructs a Sidon set by iterating through `order`.
    If `order` is None, it uses a random order.
    Returns the constructed set.
    """
    if order is None:
        order = list(range(1, N + 1))
        if rng:
            rng.shuffle(order)
        else:
            random.shuffle(order)
            
    selected = []
    sums = set()
    
    for x in order:
        # Check if x can be added
        # New sums: {x+a for a in selected} U {2x}
        # Check against existing sums
        conflict = False
        for a in selected:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
            
        if (2 * x) in sums:
            continue
            
        # Check internal conflicts of new sums (always safe as discussed above)
        # So we can add x
        selected.append(x)
        for a in selected:
            sums.add(x + a)
        sums.add(2 * x)
        
    return selected

def _max_blocking_greedy(rng, deadline):
    """
    Greedy algorithm that selects the element which 'blocks' the most other candidates.
    Blocking means: if we add x, how many other potential candidates y become invalid?
    This is a heuristic to find smaller maximal sets.
    """
    # Pre-compute potential conflicts? No, dynamic.
    # We need to simulate adding x and see how many future candidates are blocked.
    # Actually, simpler:
    # A candidate x is valid if it doesn't conflict with current sums.
    # If we pick x, we add sums {x+a for a in S} U {2x}.
    # This blocks any y such that y+a' in sums_new or 2y in sums_new for some a' in S U {x}.
    # This is expensive to compute for all y at each step.
    
    # Alternative interpretation of "max blocking":
    # Pick x that creates the most new sums that fall in the range [2, 2N].
    # Or pick x that has the most "interactions" with existing elements.
    
    # Let's stick to the proven "random order" but with a twist:
    # We will try multiple strategies.
    
    # Strategy: Randomized Greedy with "Lookahead"
    # 1. Find all valid candidates.
    # 2. For each candidate, estimate how many other candidates it will invalidate.
    # 3. Pick the one with the highest count (or random among top k).
    
    selected = []
    sums = set()
    
    # To speed up, we can maintain a list of valid candidates.
    # But checking validity is O(|S|).
    
    # Given time constraints, let's just do fast random restarts with different biases.
    # The "max blocking" insight from the other solution might be implemented as:
    # Sort candidates by a heuristic score.
    
    # Heuristic: Prefer numbers that are "central" or have specific modular properties?
    # Or just random is often best for this NP-hard-ish problem.
    
    # Let's implement a specific "greedy with random tie-breaking" that is very fast.
    
    candidates = list(range(1, N + 1))
    
    while candidates:
        if time.time() > deadline:
            break
            
        # Filter valid candidates
        valid = []
        for x in candidates:
            conflict = False
            for a in selected:
                if (x + a) in sums:
                    conflict = True
                    break
            if conflict:
                continue
            if (2 * x) in sums:
                continue
            valid.append(x)
            
        if not valid:
            break
            
        # Selection heuristic:
        # Pick randomly from valid?
        # Or pick the one that maximizes "blocking"?
        # Let's try picking the one that is farthest from existing elements?
        # Or the one that creates the most sums in the "danger zone"?
        
        # For now, random selection from valid is robust.
        # To add diversity, we can sometimes pick the "best" by a heuristic.
        
        if rng.random() < 0.5:
            x = rng.choice(valid)
        else:
            # Heuristic: pick x that maximizes sum of distances to selected?
            # Or just the first one?
            # Let's try: pick x that is in the "middle" of the range of valids?
            # Or just random is fine.
            x = rng.choice(valid)
            
        # Add x
        selected.append(x)
        for a in selected:
            sums.add(x + a)
        sums.add(2 * x)
        
        # Update candidates: remove x
        # We don't strictly need to update the list if we filter every time,
        # but removing x is good.
        candidates.remove(x)
        
    return selected

def construct():
    start_time = time.time()
    # Safety margin: 55s limit (5s buffer for overhead)
    deadline = start_time + 55.0
    
    best_set = []
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Strategy 1: Deterministic orders
    orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
        list(range(1, N + 1, 2)), # Odds
        list(range(2, N + 1, 2)), # Evens
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s = _greedy_construct(order)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
    # Strategy 2: Randomized Greedy (Standard)
    while time.time() < deadline:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s = _greedy_construct(order, rng)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            if best_size <= 13:
                break

    # Strategy 3: "Max Blocking" / Heuristic Greedy
    # We try to implement the specific insight from the inspiration.
    # We'll run several iterations of a smarter greedy.
    
    while time.time() < deadline - 5.0: # Leave 5s for final check
        # We can try to bias the random shuffle.
        # E.g., assign random weights and sort by them? That's just a shuffle.
        # Maybe sort by "density" of potential conflicts?
        
        # Let's try a different heuristic:
        # Start with a random set of size k, then try to extend/fix?
        # No, greedy is safer.
        
        # Let's try "Weighted Random" where weight is 1/x or 1/(N-x)?
        # Or just random is fine.
        
        # To ensure we explore the "max blocking" space, let's try
        # selecting the candidate that adds the most new sums to the `sums` set
        # that are within [2, 2N]. But all sums are in [2, 2N].
        # So we want to maximize the number of NEW sums added?
        # Each addition adds |S|+1 new sums.
        # So we want to add when |S| is large? No, we want to minimize |S|.
        
        # Actually, the "max blocking" heuristic usually means:
        # Pick x such that the set of y that become INVALID is maximized.
        # y becomes invalid if y+a in sums_new or 2y in sums_new.
        
        # Let's implement this specific check in a limited way for the first few steps.
        
        selected = []
        sums = set()
        candidates = list(range(1, N + 1))
        
        # Limit the "smart" selection to the first 5 elements to keep it fast?
        # Or do it always but with a time check.
        
        while candidates:
            if time.time() > deadline:
                break
                
            valid = []
            for x in candidates:
                conflict = False
                for a in selected:
                    if (x + a) in sums:
                        conflict = True
                        break
                if conflict:
                    continue
                if (2 * x) in sums:
                    continue
                valid.append(x)
                
            if not valid:
                break
                
            if len(valid) > 1 and rng.random() < 0.7:
                # Heuristic selection
                # Estimate blocking power
                best_x = None
                best_score = -1
                
                # Sample a subset of valid candidates if too many
                sample = valid if len(valid) < 20 else rng.sample(valid, 20)
                
                for x in sample:
                    # Simulate adding x
                    new_sums = set(sums)
                    for a in selected:
                        new_sums.add(x + a)
                    new_sums.add(2 * x)
                    
                    # Count how many other valid candidates are blocked
                    blocked = 0
                    for y in valid:
                        if y == x:
                            continue
                        # Check if y is blocked by new_sums
                        is_blocked = False
                        for a in selected + [x]:
                            if (y + a) in new_sums:
                                is_blocked = True
                                break
                        if not is_blocked and (2 * y) in new_sums:
                            is_blocked = True
                        if is_blocked:
                            blocked += 1
                    
                    # We want to maximize blocked?
                    # Yes, to make the set maximal faster with fewer elements.
                    if blocked > best_score:
                        best_score = blocked
                        best_x = x
                        
                if best_x is not None:
                    x = best_x
                else:
                    x = rng.choice(valid)
            else:
                x = rng.choice(valid)
                
            # Add x
            selected.append(x)
            for a in selected:
                sums.add(x + a)
            sums.add(2 * x)
            
            candidates.remove(x)
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]
            if best_size <= 13:
                break

    # Final verification
    if not _is_valid_solution(best_set):
        # Fallback
        s = _greedy_construct(list(range(1, N+1)))
        return s

    return best_set

# EVOLVE-BLOCK-END
