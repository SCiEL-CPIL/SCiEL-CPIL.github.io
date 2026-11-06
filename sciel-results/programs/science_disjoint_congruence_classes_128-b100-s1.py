import math
import random
import time

MAX_MOD = 128
DEADLINE_MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _construct_greedy(moduli_order, time_limit, rng):
    pairs = []
    # pairs is a list of (r, m)
    
    for m in moduli_order:
        if time.time() > time_limit:
            break
        
        valid_residues = []
        for r in range(m):
            ok = True
            for (r2, m2) in pairs:
                if not _disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                valid_residues.append(r)
        
        if valid_residues:
            # Pick a random valid residue
            r = rng.choice(valid_residues)
            pairs.append((r, m))
            
    return pairs

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    # Precompute GCDs for ordering heuristic
    # score[m] = sum of gcd(m, k) for k in all_mods, k != m
    scores = {}
    for m in all_mods:
        s = 0
        for k in all_mods:
            if m != k:
                s += math.gcd(m, k)
        scores[m] = s
        
    # Create different orderings
    # 1. Ascending
    order_asc = sorted(all_mods)
    # 2. Descending
    order_desc = sorted(all_mods, reverse=True)
    # 3. Descending score (most constrained first)
    order_score = sorted(all_mods, key=lambda m: scores[m], reverse=True)
    # 4. Ascending score
    order_score_asc = sorted(all_mods, key=lambda m: scores[m])
    
    best_pairs = []
    best_len = 0
    
    # Run randomized greedy multiple times
    # We will mix the orderings and random shuffles
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        rng = random.Random(num_restarts)
        
        # Choose an ordering strategy
        # 50% random shuffle, 25% score desc, 12.5% asc, 12.5% desc
        r = rng.random()
        if r < 0.5:
            order = all_mods[:]
            rng.shuffle(order)
        elif r < 0.75:
            order = order_score[:]
            # Add slight randomness to the order? No, keep it deterministic for this strategy
        elif r < 0.875:
            order = order_asc[:]
        else:
            order = order_desc[:]
            
        pairs = _construct_greedy(order, deadline, rng)
        
        if len(pairs) > best_len:
            best_len = len(pairs)
            best_pairs = pairs
            
        # Optional: Local Search / Repair
        # If we have time, try to improve the current best
        if time.time() < deadline - 10.0:
             # Try to add missed moduli
             used_mods = set(m for _, m in best_pairs)
             missed = [m for m in all_mods if m not in used_mods]
             if missed:
                 # Pick a random missed modulus
                 m_to_add = rng.choice(missed)
                 # Try to find a residue for it
                 # If it fails, we might need to re-optimize existing residues
                 # This is complex, so we'll just rely on restarts for now
                 # A simple improvement: if we find a new best, we can try to "fill in" gaps
                 pass

    return [[r, m] for r, m in best_pairs]
