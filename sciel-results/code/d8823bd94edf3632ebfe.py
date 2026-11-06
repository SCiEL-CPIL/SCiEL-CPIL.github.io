"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_DEADLINE_MARGIN = 5.0  # seconds before the 90s wall-clock kill

def _square_nonzero_count(coeffs):
    """Return (nonzero_count, length) of the exact square of a coefficient list."""
    n = len(coeffs)
    # degree of square is 2*(n-1)
    # Build square coefficients
    sq_len = 2 * n - 1
    sq = [0] * sq_len
    # Only iterate over nonzero terms of coeffs
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    for i, ci in nz:
        for j, cj in nz:
            sq[i + j] += ci * cj
    return sum(1 for x in sq if x != 0), sq_len

def _square_coeff_list(coeffs):
    """Return the full square coefficient list."""
    n = len(coeffs)
    sq_len = 2 * n - 1
    sq = [0] * sq_len
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    for i, ci in nz:
        for j, cj in nz:
            sq[i + j] += ci * cj
    return sq

def _validate(coeffs):
    """Check all constraints. Returns (valid, nz_count_in_poly)."""
    if len(coeffs) > 41:
        return False, 0
    if coeffs[-1] == 0:
        return False, 0
    for c in coeffs:
        if c < -64 or c > 64:
            return False, 0
    nz = sum(1 for c in coeffs if c != 0)
    if nz != 12:
        return False, 0
    return True, nz

def _score(coeffs):
    """Return the raw value: number of nonzero terms in the square. Lower is better."""
    valid, _ = _validate(coeffs)
    if not valid:
        return 1000  # invalid gets high penalty
    cnt, _ = _square_nonzero_count(coeffs)
    return cnt

def _random_candidate():
    """Generate a random valid candidate with exactly 12 nonzero terms."""
    while True:
        # Choose 12 distinct positions from 0..40
        positions = random.sample(range(41), 12)
        positions.sort()
        # Ensure last coefficient is nonzero: make sure position 40 is included OR the max position has nonzero
        # Actually we need coeffs[-1] != 0, so we need the highest-degree term to be nonzero.
        # Let's ensure that the last position in our list (which becomes the length) has a nonzero coeff.
        # Better: pick length L from 12 to 41, then pick 12 positions from 0..L-1, ensuring L-1 is included.
        L = random.randint(12, 41)
        # Position L-1 must be one of the nonzero terms
        remaining = random.sample(range(0, L - 1), 11)
        positions = sorted(remaining + [L - 1])
        
        coeffs = [0] * L
        for pos in positions:
            # Choose coefficient in [-64, 64], nonzero
            coeffs[pos] = random.randint(-64, 64)
            if coeffs[pos] == 0:
                coeffs[pos] = random.choice([-1, 1, -2, 2, -3, 3])
        
        valid, _ = _validate(coeffs)
        if valid:
            return coeffs

def _hill_climb(coeffs, max_steps=2000, deadline=None):
    """Local search: try small perturbations to improve the square sparsity."""
    best = list(coeffs)
    best_score = _score(best)
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        if best_score <= 6:  # very hard to beat this; 6 might be near-optimal
            break
        
        # Try modifying one coefficient
        c = list(best)
        nz_positions = [i for i, x in enumerate(c) if x != 0]
        
        # Strategy 1: change a coefficient value
        if nz_positions:
            pos = random.choice(nz_positions)
            old_val = c[pos]
            new_val = random.randint(-64, 64)
            while new_val == 0 and len(nz_positions) == 12:
                new_val = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8])
            c[pos] = new_val
            # If we changed a nonzero to zero, we have 11 nonzero terms - invalid
            if old_val != 0 and new_val == 0:
                continue
            if old_val == 0 and new_val != 0 and len(nz_positions) == 12:
                continue
            valid, _ = _validate(c)
            if valid:
                s = _score(c)
                if s < best_score:
                    best_score = s
                    best = c
            continue
        
        # Strategy 2: move a nonzero term to a new position
        nz_indices = [i for i, x in enumerate(best) if x != 0]
        if len(nz_indices) >= 2:
            move_from = random.choice(nz_indices)
            # Don't move the last nonzero if it's at the end (would change length)
            # Actually let's just try to move to a random position
            new_pos = random.randint(0, min(40, len(best) - 1))
            if new_pos == move_from:
                continue
            c = list(best)
            val = c[move_from]
            c[move_from] = 0
            c[new_pos] = val
            valid, _ = _validate(c)
            if valid:
                s = _score(c)
                if s < best_score:
                    best_score = s
                    best = c
            continue
        
        # Strategy 3: add a new nonzero term and remove another (keep count at 12)
        if len(nz_indices) == 12:
            remove_pos = random.choice(nz_indices)
            # Find a zero position to add
            zero_positions = [i for i in range(len(best)) if best[i] == 0]
            if zero_positions:
                add_pos = random.choice(zero_positions)
                c = list(best)
                val_to_move = c[remove_pos]
                c[remove_pos] = 0
                new_val = random.randint(-64, 64)
                while new_val == 0:
                    new_val = random.choice([-1, 1, -2, 2, -3, 3])
                c[add_pos] = new_val
                valid, _ = _validate(c)
                if valid:
                    s = _score(c)
                    if s < best_score:
                        best_score = s
                        best = c
                continue
    
    return best, best_score

def _try_known_structures(deadline):
    """Try some hand-crafted structures that might give good cancellation."""
    results = []
    
    # Palindromic structures
    for L in range(12, 41):
        if time.time() > deadline:
            break
        # Try symmetric coefficients
        base = [0] * L
        # Put terms at symmetric positions
        for d in range(12):
            pos = d
            if pos < L:
                base[pos] = random.choice([-1, 1, -2, 2])
            if L - 1 - d < L and L - 1 - d != pos:
                base[L - 1 - d] = base[pos] if base[pos] != 0 else random.choice([-1, 1])
        # Ensure exactly 12 nonzero
        nz = sum(1 for x in base if x != 0)
        if nz != 12:
            continue
        if base[-1] == 0:
            continue
        results.append(list(base))
    
    # Try simple patterns
    patterns = [
        [1,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,0,0,0,0,0,0,0,1,0],
        [1,1,0,0,0,0,0,0,0,0,1,1],
    ]
    for p in patterns:
        if len(p) <= 41 and p[-1] != 0:
            results.append(list(p))
    
    best = None
    best_score = 1000
    for c in results:
        valid, _ = _validate(c)
        if not valid:
            continue
        s = _score(c)
        if s < best_score:
            best_score = s
            best = c
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    random.seed(42)
    
    # Phase 1: Try known structures quickly
    best, best_score = _try_known_structures(deadline)
    if best is None:
        best = _random_candidate()
        best_score = _score(best)
    
    # Phase 2: Random restarts with hill climbing
    # Start with the best found, then try random restarts
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # With some probability, start from the best found; otherwise random
        if best is not None and random.random() < 0.3:
            # Perturb the best
            c = list(best)
            nz_positions = [i for i, x in enumerate(c) if x != 0]
            if nz_positions:
                pos = random.choice(nz_positions)
                c[pos] = random.randint(-64, 64)
                while c[pos] == 0:
                    c[pos] = random.choice([-1, 1, -2, 2, -3, 3])
            if _validate(c)[0]:
                candidate = c
            else:
                candidate = _random_candidate()
        else:
            candidate = _random_candidate()
        
        # Hill climb from this candidate
        climbed, cscore = _hill_climb(candidate, max_steps=500, deadline=deadline)
        
        if cscore < best_score:
            best_score = cscore
            best = climbed
    
    # Phase 3: Final polish on the best candidate
    if best is not None and time.time() < deadline:
        polished, pscore = _hill_climb(best, max_steps=1000, deadline=deadline)
        if pscore < best_score:
            best_score = pscore
            best = polished
    
    if best is None:
        # Fallback: return a valid candidate
        best = _random_candidate()
    
    # Ensure validity
    valid, _ = _validate(best)
    if not valid:
        best = _random_candidate()
    
    return best

# EVOLVE-BLOCK-END
