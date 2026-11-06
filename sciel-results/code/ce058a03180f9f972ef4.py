"""Combine: greedy+local-search structure (current) with bitmask compatibility,
degree-ordered randomized greedy passes, multi-start restarts, and add/sweep+swap
local search (inspiration)."""
# EVOLVE-BLOCK-START
import math
import time
import random

MAX_M = 128
MODULI = list(range(2, MAX_M + 1))
N = len(MODULI)
MOD_INDEX = {m: i for i, m in enumerate(MODULI)}

# Precompute compatibility bitmask: bit j set in mask[i] iff gcd(MODULI[i], MODULI[j]) > 1
COMPAT_MASK = [0] * N
for i in range(N):
    mi = MODULI[i]
    mask = 0
    for j in range(N):
        if i != j and math.gcd(mi, MODULI[j]) > 1:
            mask |= (1 << j)
    COMPAT_MASK[i] = mask

DEGREE = [bin(COMPAT_MASK[i]).count("1") for i in range(N)]
# Order by decreasing degree, tie-break by modulus
ORDER_BY_DEGREE = sorted(range(N), key=lambda i: (-DEGREE[i], MODULI[i]))


def _find_residue(m_idx, selected):
    """Find residue r in [0, m) for MODULI[m_idx] disjoint from `selected`
    (list of (r_i, m_i_idx)). Assumes compatibility (gcd>1) already verified.
    Returns r or None."""
    m = MODULI[m_idx]
    # For each selected class, forbid r with r % g == r_i % g where g = gcd(m, m_i)
    # Build a forbidden boolean array via stepping (cheap, m <= 128).
    forbidden = [False] * m
    for (r_i, mi_idx) in selected:
        g = math.gcd(m, MODULI[mi_idx])
        if g == 1:
            # Should not happen if compatibility checked; treat as impossible
            return None
        start = r_i % g
        for r in range(start, m, g):
            forbidden[r] = True
    for r in range(m):
        if not forbidden[r]:
            return r
    return None


def _is_compatible(m_idx, selected_mod_indices):
    """O(1) bitmask check: m_idx compatible with all selected mod indices."""
    mask = COMPAT_MASK[m_idx]
    for si in selected_mod_indices:
        if not (mask >> si) & 1:
            return False
    return True


def _greedy_pass(order_indices, selected_pairs, selected_mod_indices, used_set):
    """Continue building from current state following order_indices.
    Returns (selected_pairs, selected_mod_indices, used_set)."""
    for m_idx in order_indices:
        if m_idx in used_set:
            continue
        if not _is_compatible(m_idx, selected_mod_indices):
            continue
        r = _find_residue(m_idx, selected_pairs)
        if r is not None:
            selected_pairs.append((r, m_idx))
            selected_mod_indices.append(m_idx)
            used_set.add(m_idx)
    return selected_pairs, selected_mod_indices, used_set


def _local_search(pairs, deadline):
    """Add/sweep + swap local search. pairs: list of (r, m_idx).
    Returns best pairs found."""
    current = pairs[:]
    best = pairs[:]
    best_count = len(best)

    def snapshot(p):
        return p[:]

    while time.time() < deadline - 0.5:
        used = set(mi for (_, mi) in current)
        mod_indices = [mi for (_, mi) in current]

        # Phase 1: add + full sweep
        added = True
        while added and time.time() < deadline - 0.5:
            added = False
            for m_idx in range(N):
                if time.time() > deadline - 0.5:
                    break
                if m_idx in used:
                    continue
                if not _is_compatible(m_idx, mod_indices):
                    continue
                r = _find_residue(m_idx, current)
                if r is not None:
                    current.append((r, m_idx))
                    mod_indices.append(m_idx)
                    used.add(m_idx)
                    added = True
                    break  # restart sweep

        if len(current) > best_count:
            best_count = len(current)
            best = snapshot(current)

        # Phase 2: swap (remove 1, add 1..2)
        if not current:
            break

        rm_pos = random.randint(0, len(current) - 1)
        removed = current.pop(rm_pos)
        new_used = set(mi for (_, mi) in current)
        new_mod_indices = [mi for (_, mi) in current]

        unused = [m_idx for m_idx in range(N) if m_idx not in new_used]
        random.shuffle(unused)

        trial = current[:]
        trial_mod_indices = new_mod_indices[:]
        trial_used = new_used

        added_count = 0
        for m_idx in unused:
            if added_count >= 2:
                break
            if time.time() > deadline - 0.5:
                break
            if not _is_compatible(m_idx, trial_mod_indices):
                continue
            r = _find_residue(m_idx, trial)
            if r is not None:
                trial.append((r, m_idx))
                trial_mod_indices.append(m_idx)
                trial_used.add(m_idx)
                added_count += 1

        if added_count >= 1:
            # Try to fill remaining with a quick sweep
            if time.time() < deadline - 0.5:
                changed = True
                while changed and time.time() < deadline - 0.5:
                    changed = False
                    for m_idx in range(N):
                        if m_idx in trial_used:
                            continue
                        if not _is_compatible(m_idx, trial_mod_indices):
                            continue
                        r = _find_residue(m_idx, trial)
                        if r is not None:
                            trial.append((r, m_idx))
                            trial_mod_indices.append(m_idx)
                            trial_used.add(m_idx)
                            changed = True
                            break
            current = trial
            if len(current) > best_count:
                best_count = len(current)
                best = snapshot(current)
        else:
            # revert
            current = [removed] + current
            # (order doesn't matter for validity)

    return best


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    best_overall = []
    best_overall_count = 0

    # Multi-start: each start = randomized greedy pass + local search
    attempt = 0
    while time.time() < deadline - 2.0:
        attempt += 1
        # Build an order: degree-ordered with a random kick on a random tail
        order = list(ORDER_BY_DEGREE)
        # Randomize a random contiguous tail to diversify
        if len(order) > 4:
            cut = random.randint(2, len(order) - 2)
            tail = order[cut:]
            random.shuffle(tail)
            order = order[:cut] + tail

        # Greedy pass
        pairs, mod_indices, used = _greedy_pass(order, [], [], set())
        if not pairs:
            # fallback: pure degree order
            pairs, mod_indices, used = _greedy_pass(ORDER_BY_DEGREE, [], [], set())

        # Local search
        improved = _local_search(pairs, deadline)
        if len(improved) > best_overall_count:
            best_overall_count = len(improved)
            best_overall = improved[:]

        # Stop early if we hit a strong plateau and time is running low
        if best_overall_count >= 40 and time.time() > deadline - 8.0:
            break

    # Final polish: one more add/sweep on the best result
    if best_overall and time.time() < deadline - 0.3:
        polished = _local_search(best_overall, deadline)
        if len(polished) > best_overall_count:
            best_overall = polished
            best_overall_count = len(polished)

    # Convert (r, m_idx) -> [residue, modulus]
    return [[r, MODULI[mi]] for (r, mi) in best_overall]
# EVOLVE-BLOCK-END
