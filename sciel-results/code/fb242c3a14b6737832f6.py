"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _square_nonzero_count(coeffs):
    """Return the number of nonzero coefficients in the exact square of the polynomial."""
    n = len(coeffs)
    sq = np.zeros(2 * n - 1, dtype=object)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            cj = coeffs[j]
            if cj == 0:
                continue
            sq[i + j] += ci * cj
    return int(np.count_nonzero(sq))


def _valid_candidate(coeffs):
    """Check all constraints on the candidate coefficient list."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    if sum(1 for c in coeffs if c != 0) != 12:
        return False
    return True


def _random_candidate(rng):
    """Generate a random valid candidate."""
    # Choose degree (length-1) between 11 and 40
    degree = rng.randint(11, 40)
    length = degree + 1
    # Choose 12 distinct positions, one must be at 'degree'
    other_positions = rng.sample(range(degree), 11)
    positions = sorted(other_positions + [degree])
    coeffs = [0] * length
    for p in positions:
        coeffs[p] = rng.choice([1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8])
    return coeffs


def _try_improve_local(coeffs, rng, max_steps=2000, deadline=None):
    """Local search: try to reduce square nonzero count by small perturbations."""
    current = list(coeffs)
    current_sq = _square_nonzero_count(current)
    
    if deadline is not None and time.time() > deadline:
        return current, current_sq
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        n = len(current)
        # Try modifying one coefficient or moving a term
        op = rng.randint(0, 2)
        new_coeffs = list(current)
        
        if op == 0:
            # Change value of a nonzero term
            nz = [i for i, c in enumerate(new_coeffs) if c != 0]
            if not nz:
                continue
            idx = rng.choice(nz)
            new_coeffs[idx] = rng.randint(-64, 64)
            if new_coeffs[idx] == 0:
                continue
        elif op == 1:
            # Move a term to a new position
            nz = [i for i, c in enumerate(new_coeffs) if c != 0]
            if not nz:
                continue
            idx = rng.choice(nz)
            new_pos = rng.randint(0, 40)
            if new_pos >= len(new_coeffs):
                # Extend
                new_coeffs.extend([0] * (new_pos + 1 - len(new_coeffs)))
            val = new_coeffs[idx]
            new_coeffs[idx] = 0
            new_coeffs[new_pos] = val
            # Trim trailing zeros but keep at least length 1
            while len(new_coeffs) > 1 and new_coeffs[-1] == 0:
                new_coeffs.pop()
            if len(new_coeffs) > 41:
                continue
            if new_coeffs[-1] == 0:
                continue
        else:
            # Swap two terms
            nz = [i for i, c in enumerate(new_coeffs) if c != 0]
            if len(nz) < 2:
                continue
            i, j = rng.sample(nz, 2)
            new_coeffs[i], new_coeffs[j] = new_coeffs[j], new_coeffs[i]
        
        if not _valid_candidate(new_coeffs):
            continue
        
        new_sq = _square_nonzero_count(new_coeffs)
        if new_sq < current_sq:
            current = new_coeffs
            current_sq = new_sq
        elif new_sq == current_sq and rng.random() < 0.1:
            # Accept equal with small probability to escape plateaus
            current = new_coeffs
    
    return current, current_sq


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    rng = random.Random(42)
    
    best_coeffs = None
    best_sq = float('inf')
    
    # Phase 1: Random sampling
    while time.time() < deadline - 30.0:
        coeffs = _random_candidate(rng)
        sq = _square_nonzero_count(coeffs)
        if sq < best_sq:
            best_sq = sq
            best_coeffs = coeffs
        if best_sq <= 12:
            break
    
    # Phase 2: Local search from best and some random starts
    if best_coeffs is not None:
        c, s = _try_improve_local(best_coeffs, rng, max_steps=5000, deadline=deadline - 15.0)
        if s < best_sq:
            best_sq = s
            best_coeffs = c
    
    # Try a few more random starts with local search if time permits
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < 5:
        coeffs = _random_candidate(rng)
        c, s = _try_improve_local(coeffs, rng, max_steps=3000, deadline=deadline - 10.0)
        if s < best_sq:
            best_sq = s
            best_coeffs = c
        attempts += 1
    
    if best_coeffs is None:
        return [1] * 12
    
    return best_coeffs

# EVOLVE-BLOCK-END
