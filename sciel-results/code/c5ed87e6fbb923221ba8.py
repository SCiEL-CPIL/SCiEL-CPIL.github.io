"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

N = 256
DEADLINE = 85.0  # safety margin before 90s kill

def _compute_discrepancy(signs):
    """Compute max absolute signed sum over all APs in [1, N]."""
    n = len(signs)
    max_disc = 0
    # For each start a and step d, compute running sums
    for a in range(n):
        # step 1
        s = 0
        for b in range(a, n):
            s += signs[b]
            if abs(s) > max_disc:
                max_disc = abs(s)
        # steps > 1
        for d in range(2, n - a + 1):
            s = 0
            for b in range(a, n, d):
                s += signs[b]
                if abs(s) > max_disc:
                    max_disc = abs(s)
    return max_disc

def _evaluate_and_improve(signs, deadline):
    """Local search: flip single signs or pairs to reduce max discrepancy."""
    signs = list(signs)
    n = len(signs)
    
    # Precompute all APs as lists of indices
    aps = []
    for a in range(n):
        s = 0
        idx_list = []
        for b in range(a, n):
            idx_list.append(b)
            aps.append(idx_list[:])
        for d in range(2, n - a + 1):
            idx_list = []
            for b in range(a, n, d):
                idx_list.append(b)
            if len(idx_list) >= 2:
                aps.append(idx_list)
    
    # Compute initial sums
    sums = [0] * len(aps)
    for i, ap in enumerate(aps):
        s = 0
        for idx in ap:
            s += signs[idx]
        sums[i] = s
    
    max_disc = max(abs(s) for s in sums)
    best_signs = signs[:]
    best_max = max_disc
    
    # Build index: for each position, which APs contain it
    pos_to_aps = defaultdict(list)
    for i, ap in enumerate(aps):
        for idx in ap:
            pos_to_aps[idx].append(i)
    
    t0 = time.time()
    no_improve_count = 0
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Find the AP with max absolute sum
        max_idx = 0
        max_abs = abs(sums[0])
        for i in range(1, len(sums)):
            ai = abs(sums[i])
            if ai > max_abs:
                max_abs = ai
                max_idx = i
        
        if max_abs <= 1:
            break
        
        # Try flipping a sign in the worst AP
        ap = aps[max_idx]
        # Prefer flipping a sign that reduces the sum
        target_sign = -1 if sums[max_idx] > 0 else 1
        candidates = [idx for idx in ap if signs[idx] == target_sign]
        if not candidates:
            candidates = ap[:]
        
        rng.shuffle(candidates)
        improved = False
        for idx in candidates:
            # Flip sign at idx
            old_sign = signs[idx]
            signs[idx] = -old_sign
            # Update sums for all APs containing idx
            for ai in pos_to_aps[idx]:
                sums[ai] += (-old_sign - old_sign)  # change is -2*old_sign
            
            new_max = 0
            for s in sums:
                a = abs(s)
                if a > new_max:
                    new_max = a
            
            if new_max < max_disc:
                max_disc = new_max
                best_signs = signs[:]
                best_max = max_disc
                improved = True
                no_improve_count = 0
            else:
                # Revert
                signs[idx] = old_sign
                for ai in pos_to_aps[idx]:
                    sums[ai] += (old_sign - (-old_sign))  # revert change
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                # Random restart with small perturbation
                for _ in range(3):
                    idx = rng.randrange(n)
                    signs[idx] = -signs[idx]
                    for ai in pos_to_aps[idx]:
                        sums[ai] += -2 * signs[idx]
                no_improve_count = 0
    
    return best_signs, best_max

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    
    # Initial random signs
    rng = random.Random(171)
    signs = [rng.choice((-1, 1)) for _ in range(N)]
    
    best_signs = signs[:]
    best_max = _compute_discrepancy(signs)
    
    # Multi-start local search
    start_count = 0
    while time.time() < deadline - 5:
        start_count += 1
        if start_count == 1:
            cur_signs = signs[:]
        else:
            # Perturb best or random
            if rng.random() < 0.3:
                cur_signs = best_signs[:]
            else:
                cur_signs = [rng.choice((-1, 1)) for _ in range(N)]
            # Small perturbation
            for _ in range(rng.randint(1, 8)):
                idx = rng.randrange(N)
                cur_signs[idx] = -cur_signs[idx]
        
        improved_signs, improved_max = _evaluate_and_improve(cur_signs, deadline - 2)
        if improved_max < best_max:
            best_max = improved_max
            best_signs = improved_signs[:]
        
        if best_max <= 3:
            break
    
    return best_signs
# EVOLVE-BLOCK-END
