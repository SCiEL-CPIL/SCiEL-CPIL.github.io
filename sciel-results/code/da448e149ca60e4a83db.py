import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_score = 0.0
    best_classes = []
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    
    # We will try to find a good set of classes.
    # Strategy: Greedy with randomized restarts.
    # In each restart, we iterate through moduli in increasing order (to prioritize small m).
    # For each m, we pick a valid residue r.
    # To vary the search, we can randomize the order in which we try residues r for a given m.
    # Or, we can try different "bias" functions for residue selection.
    
    # A simple valid residue check:
    # New (m, r) is valid with existing (m_i, r_i) if r != r_i mod gcd(m, m_i) for all i.
    
    def is_valid(classes, m, r):
        for m_i, r_i in classes:
            g = gcd(m, m_i)
            if r % g == r_i % g:
                return False
        return True
    
    def greedy_solve(random_seed):
        random.seed(random_seed)
        classes = []
        # Sort moduli by value (ascending) to prioritize high 1/m
        for m in moduli:
            # Try residues in a randomized order to avoid deterministic traps
            # However, small residues are often "safer" or more likely to be valid if we've been careful.
            # Let's try residues 0..m-1 in a random order, but keep track of the best (smallest? no, just any valid).
            # Actually, the specific residue value doesn't affect the score, only validity.
            # So we just need ANY valid residue.
            # Randomizing the order of r trials might lead to different sets of classes.
            
            # Optimization: Pre-compute gcds? No, m is fixed in loop, m_i changes.
            # Let's just try residues.
            
            # To make it faster, we can try residues 0..m-1.
            # If no residue is valid, we skip this modulus.
            
            # Randomized restart idea:
            # Sometimes picking the *first* valid residue in sorted order (0, 1, ...) is good.
            # Sometimes picking a random valid residue is good.
            # Let's mix: with prob 0.5, try 0..m-1 sorted. With prob 0.5, try random order.
            
            if random.random() < 0.5:
                r_order = list(range(m))
            else:
                r_order = list(range(m))
                random.shuffle(r_order)
            
            found = False
            for r in r_order:
                if is_valid(classes, m, r):
                    classes.append([m, r])
                    found = True
                    break
            
            # If not found, we skip m.
            # Note: We don't backtrack.
        
        # Calculate score
        score = sum(1.0 / m for m, r in classes)
        return score, classes

    # Run multiple restarts
    # Estimate time per restart:
    # 180 moduli * up to 200 residues * 180 existing classes * gcd + mod.
    # ~ 180 * 200 * 180 = 6.5M operations. Very fast in Python if optimized, but pure Python loops are slow.
    # Let's optimize is_valid.
    
    def is_valid_fast(classes, m, r):
        # classes is list of [m_i, r_i]
        # We can precompute nothing easily as m changes.
        for m_i, r_i in classes:
            g = gcd(m, m_i)
            if (r - r_i) % g == 0:
                return False
        return True

    # Let's refine the greedy loop to be faster.
    def greedy_solve_fast(random_seed):
        random.seed(random_seed)
        classes = []
        for m in moduli:
            # Determine residue order
            if random.random() < 0.5:
                r_order = range(m)
            else:
                r_order = list(range(m))
                random.shuffle(r_order)
            
            # Try to find a valid r
            valid_r = -1
            for r in r_order:
                if is_valid_fast(classes, m, r):
                    valid_r = r
                    break
            
            if valid_r != -1:
                classes.append([m, valid_r])
        
        score = sum(1.0 / m for m, r in classes)
        return score, classes

    # How many restarts can we do?
    # Let's test one restart time.
    # If it's < 0.1s, we can do 1000 restarts.
    
    # Initial run
    score, classes = greedy_solve_fast(0)
    best_score = score
    best_classes = classes
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 0:
        time_per_run = elapsed
    else:
        time_per_run = 0.01 # fallback
    
    # Run more restarts until deadline
    run_count = 1
    while time.time() < deadline:
        seed = run_count
        score, classes = greedy_solve_fast(seed)
        if score > best_score:
            best_score = score
            best_classes = classes
        run_count += 1
        
        # Optional: Adaptive restart rate
        # If we are improving, continue. If stuck, maybe try different strategy?
        # For now, simple randomized restarts.
        
    return best_classes

