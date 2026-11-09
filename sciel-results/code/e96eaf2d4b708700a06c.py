"""Combined approach: Bitset infrastructure + Coverage Heuristic Greedy + Aggressive Pruning."""
import time
import random
import numpy as np

N = 16
TOTAL_POINTS = N * N
POINTS = [(x, y) for x in range(N) for y in range(N)]

# Precompute distance squared matrix
_dist2 = np.zeros((TOTAL_POINTS, TOTAL_POINTS), dtype=np.int32)
for i in range(TOTAL_POINTS):
    for j in range(TOTAL_POINTS):
        dx = POINTS[i][0] - POINTS[j][0]
        dy = POINTS[i][1] - POINTS[j][1]
        _dist2[i, j] = dx * dx + dy * dy

# Precompute isosceles relations:
# _iso_mask[p][a] is a bitset of points q such that {p, a, q} is isosceles.
# Note: _iso_mask[p][a] == _iso_mask[a][p] == _iso_mask[q][p] & _iso_mask[q][a] logic is symmetric in the triangle.
# Specifically, if p, a, q is isosceles, then p is in _iso_mask[a][q], a is in _iso_mask[p][q], q is in _iso_mask[p][a].
_iso_mask = []
for p in range(TOTAL_POINTS):
    row = [0] * TOTAL_POINTS
    for a in range(TOTAL_POINTS):
        if a == p:
            continue
        mask = 0
        dpa = _dist2[p, a]
        for q in range(TOTAL_POINTS):
            if q == p or q == a:
                continue
            dpq = _dist2[p, q]
            daq = _dist2[a, q]
            if dpa == dpq or dpa == daq or dpq == daq:
                mask |= (1 << q)
        row[a] = mask
    _iso_mask.append(row)

ALL_MASK = (1 << TOTAL_POINTS) - 1

def _can_add(p, S_mask):
    """Check if point p can be added to set S_mask without creating isosceles."""
    if (S_mask >> p) & 1:
        return False
    im = _iso_mask[p]
    S_temp = S_mask
    while S_temp:
        lsb = S_temp & -S_temp
        a = lsb.bit_length() - 1
        S_temp ^= lsb
        # If any point in S is in im[a] (which are points forming iso with p and a), then p cannot be added
        # Wait, im[p][a] contains q such that p, a, q is iso.
        # So if S_mask & im[p][a] is non-zero, there is a q in S such that p, a, q is iso.
        # Thus p cannot be added.
        if S_mask & im[a]:
            return False
    return True

def _get_valid_points(S_mask):
    """Return bitset of points not in S_mask that can be added to S_mask."""
    valid = ALL_MASK ^ S_mask
    # We need to filter valid points.
    # A point p is valid if for all a in S, S_mask & _iso_mask[p][a] == 0.
    # This check is expensive to do for all p in a loop if we do it naively.
    # However, we can just call _can_add for each p in valid.
    # Given N=16, 256 points, this is fast enough.
    result = 0
    v_temp = valid
    while v_temp:
        lsb = v_temp & -v_temp
        p = lsb.bit_length() - 1
        v_temp ^= lsb
        if _can_add(p, S_mask):
            result |= lsb
    return result

def _greedy_construct_coverage():
    """Build a maximal no-isosceles set using coverage heuristic."""
    S_mask = 0
    while True:
        valid_mask = _get_valid_points(S_mask)
        if not valid_mask:
            break
        
        # Calculate coverage for each valid point
        best_p = -1
        best_cov = -1
        v_temp = valid_mask
        while v_temp:
            lsb = v_temp & -v_temp
            p = lsb.bit_length() - 1
            v_temp ^= lsb
            
            # Calculate how many valid points are blocked by adding p
            blocked = 0
            S_temp = S_mask
            while S_temp:
                lsb_s = S_temp & -S_temp
                a = lsb_s.bit_length() - 1
                S_temp ^= lsb_s
                blocked |= _iso_mask[p][a]
            
            # Count valid points that are in blocked set
            cov = bin(valid_mask & blocked).count('1')
            
            # Add some randomness to tie-breaking
            if cov > best_cov or (cov == best_cov and random.random() < 0.5):
                best_cov = cov
                best_p = p
        
        if best_p == -1:
            break
            
        S_mask |= (1 << best_p)
        
    return S_mask

def _is_maximal(S_mask):
    """Check if S_mask is maximal: no point outside can be added."""
    not_S = ALL_MASK ^ S_mask
    p_bit = not_S
    while p_bit:
        lsb = p_bit & -p_bit
        p = lsb.bit_length() - 1
        p_bit ^= lsb
        if _can_add(p, S_mask):
            return False
    return True

def _prune(S_mask):
    """Try to remove points while maintaining maximality. Returns pruned mask."""
    improved = True
    while improved:
        improved = False
        S_temp = S_mask
        while S_temp:
            lsb = S_temp & -S_temp
            p = lsb.bit_length() - 1
            S_temp ^= lsb
            new_S = S_mask ^ lsb
            if _is_maximal(new_S):
                S_mask = new_S
                improved = True
                break
    return S_mask

def _mask_to_list(S_mask):
    points = []
    m = S_mask
    while m:
        lsb = m & -m
        p = lsb.bit_length() - 1
        points.append(POINTS[p])
        m ^= lsb
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 285.0

    best_mask = 0
    best_size = TOTAL_POINTS + 1

    # Phase 1: Multiple greedy constructions with coverage heuristic
    num_trials = 150 # Reduced slightly to allow for more pruning time if needed, but coverage is O(N^3) roughly per trial
    
    for trial in range(num_trials):
        if time.time() > deadline - 30:
            break
        
        # Randomize the tie-breaking in coverage heuristic by shuffling the valid points iteration?
        # The current _greedy_construct_coverage uses random.random() for ties.
        # We can also just run it multiple times.
        
        S_mask = _greedy_construct_coverage()
        size = bin(S_mask).count('1')

        if size < best_size:
            best_size = size
            best_mask = S_mask

        # Try to prune
        pruned = _prune(S_mask)
        pruned_size = bin(pruned).count('1')
        if pruned_size < best_size:
            best_size = pruned_size
            best_mask = pruned

    # Phase 2: Local search on best_mask
    if time.time() < deadline - 20:
        current = best_mask
        current_size = best_size
        no_improve_count = 0

        while time.time() < deadline - 10 and no_improve_count < 50:
            # Try removing a point
            found = False
            S_temp = current
            pts_in_S = []
            m = S_temp
            while m:
                lsb = m & -m
                p = lsb.bit_length() - 1
                pts_in_S.append(p)
                m ^= lsb
            random.shuffle(pts_in_S)

            for p in pts_in_S:
                new_S = current ^ (1 << p)
                if _is_maximal(new_S):
                    current = new_S
                    current_size -= 1
                    found = True
                    break

            if found:
                no_improve_count = 0
                continue

            # Try swap: remove one, add another
            swap_found = False
            for p in pts_in_S:
                if time.time() > deadline - 10:
                    break
                S_without = current ^ (1 << p)
                # Find addable points to S_without
                not_S = ALL_MASK ^ S_without
                cand_bit = not_S
                candidates = []
                while cand_bit:
                    lsb = cand_bit & -cand_bit
                    q = lsb.bit_length() - 1
                    cand_bit ^= lsb
                    if _can_add(q, S_without):
                        candidates.append(q)

                if not candidates:
                    continue

                for q in random.sample(candidates, min(len(candidates), 10)):
                    new_S = S_without | (1 << q)
                    if _is_maximal(new_S):
                        current = new_S
                        swap_found = True
                        break
                if swap_found:
                    break

            if not swap_found:
                no_improve_count += 1
            else:
                no_improve_count = 0

        # Final prune
        pruned = _prune(current)
        pruned_size = bin(pruned).count('1')
        if pruned_size < best_size:
            best_size = pruned_size
            best_mask = pruned

    # Phase 3: If we have time, try more aggressive random restarts with pruning
    if time.time() < deadline - 15:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            S_mask = _greedy_construct_coverage()
            pruned = _prune(S_mask)
            sz = bin(pruned).count('1')
            if sz < best_size:
                best_size = sz
                best_mask = pruned

    return _mask_to_list(best_mask)
