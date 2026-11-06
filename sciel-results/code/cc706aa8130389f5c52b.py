import time
import random

T_START = time.time()
DEADLINE = 75.0  # Safety margin

def _square_nonzero_count(coeffs):
    """Return number of nonzero coefficients in the exact square of coeff list."""
    n = len(coeffs)
    if n == 0:
        return 0
    nz_indices = [(i, coeffs[i]) for i in range(n) if coeffs[i] != 0]
    nz_count = len(nz_indices)
    sq = [0] * (2 * n - 1)
    for i_idx, a in nz_indices:
        ai2 = a * a
        sq[2 * i_idx] += ai2
        for j_idx, b in nz_indices:
            if j_idx > i_idx:
                sq[i_idx + j_idx] += 2 * a * b
    count = 0
    for v in sq:
        if v != 0:
            count += 1
    return count

def _valid_candidate(coeffs):
    """Check all constraints on the candidate list."""
    if not isinstance(coeffs, list):
        return False
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    for v in coeffs:
        if v < -64 or v > 64:
            return False
    nnz = sum(1 for v in coeffs if v != 0)
    if nnz != 12:
        return False
    return True

def _try_polynomial(coeffs):
    """If valid, return (count, coeffs); else (None, None)."""
    if not _valid_candidate(coeffs):
        return None, None
    return _square_nonzero_count(coeffs), coeffs

def construct():
    best_count = 100
    best_coeffs = None
    end_time = T_START + DEADLINE

    def expired():
        return time.time() > end_time

    def build_poly(positions, values):
        coeffs = [0] * 41
        for p, v in zip(positions, values):
            if 0 <= p < 41:
                coeffs[p] = v
            else:
                return None
        k = 40
        while k >= 0 and coeffs[k] == 0:
            k -= 1
        if k < 0:
            return None
        return coeffs[:k+1]

    def update_best(coeffs):
        nonlocal best_count, best_coeffs
        if coeffs is None:
            return
        c, _ = _try_polynomial(coeffs)
        if c is not None and c < best_count:
            best_count = c
            best_coeffs = list(coeffs)

    # --- Phase 1: Structured Position Families ---
    position_sets = []

    # Consecutive blocks
    for start in range(0, 30):
        position_sets.append(list(range(start, start + 12)))

    # Arithmetic progressions (gaps 1-5)
    for gap in range(1, 6):
        for start in range(0, 41 - 11*gap + 1):
            positions = [start + i * gap for i in range(12)]
            position_sets.append(positions)

    # Two-gap patterns
    for g1 in range(1, 6):
        for g2 in range(1, 6):
            for start in range(0, 30):
                pos = [start + i * g1 for i in range(6)]
                mid = pos[-1] + 1
                pos += [mid + i * g2 for i in range(6)]
                if max(pos) <= 40:
                    position_sets.append(pos)

    # Symmetric about center
    for center in range(20, 30):
        for radius in range(1, 15):
            positions = []
            for k in range(1, 10):
                if len(positions) >= 12: break
                if center - k >= 0:
                    positions.append(center - k)
                if center + k < 41:
                    positions.append(center + k)
            if len(positions) == 12:
                positions.sort()
                position_sets.append(positions)

    # Powers of 2
    p2 = [1, 2, 4, 8, 16, 32]
    for offset in range(0, 20):
        base = [offset + x for x in p2]
        if max(base) <= 40:
            position_sets.append(base)

    # Additive-energy-guided: positions with many sum collisions
    # Try specific known sparse structures
    # e.g., positions that form a difference set
    # For 12 terms, a (12, k, lambda) difference set would be ideal
    # We'll generate some candidates via random but biased towards clusters

    # Deduplicate
    seen = set()
    unique_pos = []
    for ps in position_sets:
        key = tuple(sorted(ps))
        if key not in seen:
            seen.add(key)
            unique_pos.append(sorted(ps))
    position_sets = unique_pos

    # Sign patterns to try
    sign_patterns = [
        [1]*12,
        [1 if i%2==0 else -1 for i in range(12)],
        [1]*6 + [-1]*6,
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
    ]

    for ps in position_sets:
        if expired():
            break
        for signs in sign_patterns:
            if expired():
                break
            update_best(build_poly(ps, signs))

    # --- Phase 2: Local Search with Sign/Position Moves ---
    def local_search_fixed(start_coeffs, max_steps=3000):
        nonlocal best_count, best_coeffs
        if start_coeffs is None:
            return
        
        # Convert to 41-length
        arr = [0] * 41
        for i, v in enumerate(start_coeffs):
            if i < 41:
                arr[i] = v
                
        curr_arr = list(arr)
        # Trim for evaluation
        def get_trimmed(a):
            k = 40
            while k >= 0 and a[k] == 0:
                k -= 1
            if k < 0: return None
            return a[:k+1]
            
        curr_trimmed = get_trimmed(curr_arr)
        curr_count, _ = _try_polynomial(curr_trimmed)
        if curr_count is None:
            return
        
        # Track recent moves to avoid cycling
        recent_states = set()
        no_improve_count = 0
        
        for step in range(max_steps):
            if expired():
                return
            
            new_arr = list(curr_arr)
            move_type = random.randint(0, 4)
            
            if move_type == 0:
                # Flip sign
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                new_arr[idx] = -new_arr[idx]
            elif move_type == 1:
                # Change coefficient magnitude
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                old = new_arr[idx]
                # Try small changes
                choices = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
                choices = [c for c in choices if c != old and abs(c) <= 64]
                if choices:
                    new_arr[idx] = random.choice(choices)
            elif move_type == 2:
                # Shift position
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_idx = idx + shift
                if 0 <= new_idx < 41 and new_idx != idx:
                    if new_arr[new_idx] == 0:
                        new_arr[new_idx] = new_arr[idx]
                        new_arr[idx] = 0
                    else:
                        continue
            elif move_type == 3:
                # Swap two positions
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if len(nz) < 2: continue
                idx1, idx2 = random.sample(nz, 2)
                new_arr[idx1], new_arr[idx2] = new_arr[idx2], new_arr[idx1]
            else:
                # Move one term to a random new position
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                val = new_arr[idx]
                empty = [i for i in range(41) if new_arr[i] == 0]
                if empty:
                    new_idx = random.choice(empty)
                    new_arr[new_idx] = val
                    new_arr[idx] = 0

            new_trimmed = get_trimmed(new_arr)
            if new_trimmed is None:
                continue
            c, _ = _try_polynomial(new_trimmed)
            if c is not None:
                state_key = tuple(new_arr)
                if state_key in recent_states:
                    continue
                recent_states.add(state_key)
                if len(recent_states) > 100:
                    recent_states = set(list(recent_states)[-50:])
                
                if c < curr_count:
                    curr_count = c
                    curr_arr = list(new_arr)
                    no_improve_count = 0
                    if c < best_count:
                        best_count = c
                        best_coeffs = list(new_trimmed)
                else:
                    no_improve_count += 1
                    if no_improve_count > 200:
                        break

    # Run local search from best found so far
    if best_coeffs:
        local_search_fixed(best_coeffs)

    # Run local search from a few random starts
    random.seed(42)
    for _ in range(15):
        if expired():
            break
        # Generate positions with some structure
        base = random.randint(0, 29)
        gaps = [random.randint(1, 4) for _ in range(11)]
        positions = [base]
        for g in gaps:
            positions.append(positions[-1] + g)
        if max(positions) >= 41:
            positions = random.sample(range(41), 12)
        positions.sort()
        values = [random.choice([1, -1]) for _ in range(12)]
        start_c = build_poly(positions, values)
        if start_c:
            local_search_fixed(start_c, max_steps=800)

    # Final polish: try all sign combinations for the best position set if time permits
    # This is expensive, so only if best_coeffs is found and time is left
    if best_coeffs and not expired():
        # Extract positions
        pos = [i for i, v in enumerate(best_coeffs) if v != 0]
        if len(pos) == 12:
            # Try all 2^12 = 4096 sign combinations
            for mask in range(4096):
                if expired(): break
                signs = []
                for i in range(12):
                    signs.append(-1 if (mask >> i) & 1 else 1)
                update_best(build_poly(pos, signs))

    if best_coeffs is None:
        best_coeffs = [1] * 12

    return best_coeffs
