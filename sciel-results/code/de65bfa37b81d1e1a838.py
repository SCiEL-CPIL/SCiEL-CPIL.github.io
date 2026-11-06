import random
import time

def construct():
    n = 128
    target_lo = 128
    target_hi = 256
    targets = range(target_lo, target_hi + 1)
    num_targets = target_hi - target_lo + 1
    
    # Helper to compute reps from scratch
    def compute_reps(A_list):
        reps = [0] * num_targets
        # Sort for consistent iteration, though not strictly needed for count
        for i in range(len(A_list)):
            ai = A_list[i]
            # Pair (ai, ai)
            s = ai + ai
            if target_lo <= s <= target_hi:
                reps[s - target_lo] += 1
            # Pairs (ai, aj) for j > i
            for j in range(i + 1, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if target_lo <= s <= target_hi:
                    reps[s - target_lo] += 1
        return reps

    def get_metrics(reps):
        peak = max(reps) if reps else 0
        uncovered = sum(1 for r in reps if r == 0)
        return peak, uncovered

    # Initialize A
    # Start with a reasonably sized random set
    # Size 40 is likely to cover everything
    A_set = set()
    # Ensure 0 and 128 are in A to cover extremes? 
    # 0+128=128. 128+128=256.
    # If we don't have 128, we can't make 256. So 128 MUST be in A.
    # If we don't have 0, can we make 128? Yes, e.g., 64+64.
    # But 128 is critical for 256.
    A_set.add(128)
    
    # Add random elements
    candidates = list(range(0, 128)) # 0 to 127
    random.shuffle(candidates)
    for c in candidates[:30]:
        A_set.add(c)
    
    A_list = list(A_set)
    
    # Ensure coverage initially by adding more if needed
    reps = compute_reps(A_list)
    peak, uncovered = get_metrics(reps)
    while uncovered > 0:
        # Find an uncovered target
        uncovered_targets = [i + target_lo for i, r in enumerate(reps) if r == 0]
        # Pick a random uncovered target and try to fix it by adding an element
        # Simple strategy: add random elements until covered
        # This might increase peak, but we'll optimize later
        added = False
        for c in candidates:
            if c not in A_set:
                A_set.add(c)
                A_list.append(c)
                reps = compute_reps(A_list)
                peak, uncovered = get_metrics(reps)
                added = True
                break
        if not added:
            # If no candidates left, we have all 129 elements, definitely covered
            break

    # Simulated Annealing
    start_time = time.time()
    current_peak, current_uncovered = get_metrics(reps)
    current_cost = current_peak * 10 + current_uncovered * 1000
    
    best_peak, best_uncovered = current_peak, current_uncovered
    best_cost = current_cost
    best_A = set(A_set)
    
    T = 10.0
    cooling = 0.999
    
    while T > 0.01 and time.time() - start_time < 50:
        # Generate a move
        move_type = random.choice(['swap', 'add', 'remove'])
        
        new_A_set = set(A_set)
        new_reps = reps[:] # Copy
        
        if move_type == 'swap':
            if len(A_set) < 2:
                continue
            # Pick a to remove, b to add
            a = random.choice(list(A_set))
            # Pick b from [0, 128] not in A_set
            # To be efficient, pre-generate candidates or pick from range
            b = random.randint(0, 128)
            if b in A_set:
                continue
            
            # Update reps: remove a, add b
            # This is O(|A|)
            # We need to be careful with the incremental update
            
            # Let's just recompute for simplicity first, then optimize if too slow.
            # Recompute is O(|A|^2) ~ 900.
            # 100k iters * 900 = 90M. Might be tight in Python.
            # Let's try incremental.
            
            # Incremental update logic:
            # 1. Remove contributions of 'a'
            #    a is in A_set.
            #    Pairs involving a: (a, x) for x in A_set, x != a. And (a, a).
            #    Note: In our compute_reps, we counted (ai, aj) for i<j.
            #    So we need to decrement for all x in A_set such that x > a?
            #    Or simply: for each x in A_set:
            #       if x == a: decrement (a+a)
            #       if x != a: 
            #          The pair (min(a,x), max(a,x)) was counted once.
            #          Decrement reps[min(a,x)+max(a,x)]
            
            # This is getting complex to manage state correctly with a copy.
            # Given 60s limit, and Python speed, let's stick to recompute but limit iterations or optimize inner loop.
            # Actually, let's just recompute. 10k iterations might be enough if we are smart.
            # Or use NumPy?
            # Let's try a hybrid: Recompute, but limit time.
            
            A_set.remove(a)
            A_set.add(b)
            A_list = list(A_set)
            new_reps = compute_reps(A_list)
            new_peak, new_uncovered = get_metrics(new_reps)
            new_cost = new_peak * 10 + new_uncovered * 1000
            
        elif move_type == 'add':
            b = random.randint(0, 128)
            if b in A_set:
                continue
            A_set.add(b)
            A_list = list(A_set)
            new_reps = compute_reps(A_list)
            new_peak, new_uncovered = get_metrics(new_reps)
            new_cost = new_peak * 10 + new_uncovered * 1000
            
        elif move_type == 'remove':
            if len(A_set) <= 5: # Keep minimal size
                continue
            a = random.choice(list(A_set))
            # Do not remove 128 if it's the only way to get 256? 
            # If we remove 128, we can't get 256. So 128 must stay.
            if a == 128:
                continue
            A_set.remove(a)
            A_list = list(A_set)
            new_reps = compute_reps(A_list)
            new_peak, new_uncovered = get_metrics(new_reps)
            new_cost = new_peak * 10 + new_uncovered * 1000
            
        # Acceptance
        if new_cost < current_cost or (new_cost > current_cost and random.random() < pow(2.71828, -(new_cost - current_cost) / T)):
            # Accept
            current_cost = new_cost
            current_peak = new_peak
            current_uncovered = new_uncovered
            reps = new_reps
            A_list = list(A_set) # A_set already updated
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_peak = current_peak
                best_uncovered = current_uncovered
                best_A = set(A_set)
                
        else:
            # Reject: Revert A_set
            if move_type == 'swap':
                A_set.remove(b)
                A_set.add(a)
            elif move_type == 'add':
                A_set.remove(b)
            elif move_type == 'remove':
                A_set.add(a)
            
        T *= cooling

    return sorted(best_A)
