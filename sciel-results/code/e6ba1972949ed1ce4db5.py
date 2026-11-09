"""Combined approach: bitset precomputation for speed + coverage heuristic + aggressive pruning."""
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
# iso_mask[p] is a list of TOTAL_POINTS integers (bitsets)
# iso_mask[p][i] has bit j set iff {p, i, j} forms an isosceles triangle
# (and p != i != j != p)
_iso_mask = []
for p in range(TOTAL_POINTS):
    row = [0] * TOTAL_POINTS
    for i in range(TOTAL_POINTS):
        if i == p:
            continue
        mask = 0
        dpi = _dist2[p, i]
        for j in range(TOTAL_POINTS):
            if j == p or j == i:
                continue
            dpj = _dist2[p, j]
            dij = _dist2[i, j]
            if dpi == dpj or dpi == dij or dpj == dij:
                mask |= (1 << j)
        row[i] = mask
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
        i = lsb.bit_length() - 1
        S_temp ^= lsb
        if S_mask & im[i]:
            return False
    return True


def _is_maximal(S_mask):
    """Check if S_mask is maximal: no point outside can be added."""
    not_S = ALL_MASK ^ S_mask
    p_bit = not_S
    while p_bit:
        lsb = p_bit & -p_bit
        p = lsb.bit_length() - 1
        p_bit ^= lsb
        im = _iso_mask[p]
        S_temp = S_mask
        blocked = False
        while S_temp:
            i_lsb = S_temp & -S_temp
            i = i_lsb.bit_length() - 1
            S_temp ^= i_lsb
            if S_mask & im[i]:
                blocked = True
                break
        if not blocked:
            return False
    return True


def _greedy_construct(order):
    """Build a maximal no-isosceles set by greedily adding points in given order."""
    S_mask = 0
    for p in order:
        if _can_add(p, S_mask):
            S_mask |= (1 << p)
    return S_mask


def _prune(S_mask):
    """Try to remove points while maintaining maximality. Returns pruned mask."""
    improved = True
    while improved:
        improved = False
        S_temp = S_mask
        while S_temp:
            lsb = S_temp & -S_temp
            i = lsb.bit_length() - 1
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

    # Phase 1: Multiple greedy constructions with different orderings
    # Use a mix of random shuffles and heuristic orderings
    num_trials = 200

    for trial in range(num_trials):
        if time.time() > deadline - 30:
            break

        if trial == 0:
            order = list(range(TOTAL_POINTS))
        elif trial % 3 == 1:
            order = list(range(TOTAL_POINTS))
            random.shuffle(order)
        else:
            # Heuristic: order points by how "central" they are (more isosceles relations = more blocking)
            # Points that block more are added earlier in greedy to create denser blocking
            scores = []
            for p in range(TOTAL_POINTS):
                cnt = 0
                im = _iso_mask[p]
                for i in range(TOTAL_POINTS):
                    if i != p:
                        cnt += bin(im[i]).count('1')
                scores.append((random.random(), cnt, p))
            scores.sort(key=lambda x: (-x[1], x[0]))
            order = [s[2] for s in scores]

        S_mask = _greedy_construct(order)
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

    # Phase 2: Local search / swap-based improvement on best_mask
    if time.time() < deadline - 20:
        current = best_mask
        current_size = best_size
        no_improve_count = 0

        while time.time() < deadline - 10 and no_improve_count < 50:
            # Try removing a point
            found = False
            S_temp = current
            # Shuffle for diversity
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

            # Try swap: remove one, add another (net zero size change, but may enable future removals)
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
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            order = list(range(TOTAL_POINTS))
            random.shuffle(order)
            S_mask = _greedy_construct(order)
            pruned = _prune(S_mask)
            sz = bin(pruned).count('1')
            if sz < best_size:
                best_size = sz
                best_mask = pruned

    return _mask_to_list(best_mask)
