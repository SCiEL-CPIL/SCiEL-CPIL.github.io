"""Size-efficient sorting network on 18 wires.

Design (fresh):
  * A sorting network == a topological order of comparators (if i<j, (i,j) may
    precede (j,k)).  We therefore represent a candidate as a *permutation* of a
    fixed comparator set (the seed odd-even transposition network, 153 comps)
    and evaluate it in that order.
  * Correctness is certified by the 0/1 principle over all 2^18 Boolean inputs,
    vectorised with NumPy (one pass over all 262144 masks).
  * Zero-based redundancy removal: a comparator that never changes any Boolean
    output can be dropped.  This is sound (0/1 principle) and yields a strictly
    smaller network.
  * Local search: swap adjacent comparators in the order; keep the swap only if
    it preserves sorting AND does not increase the comparator count.  This
    minimises size while never leaving the "valid" region.
  * A hard wall-clock deadline guard returns the best valid result early.
"""

import time
import numpy as np

N = 18
N_COMBOS = N * (N - 1) // 2          # 153
DEADLINE = 78.0                        # seconds (safety margin under 90s)
SEED = 20240607

# ---------------------------------------------------------------------------
# Seed comparator set (odd-even transposition network) as a list of pairs.
# ---------------------------------------------------------------------------
_SEED_PAIRS = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]


def _pair_to_idx(p):
    a, b = p
    return a * N - a * (a + 1) // 2 + (b - a - 1)


# Precompute, once at import time, the (i, j) for every comparator index.
_PAIR_I = np.zeros(N_COMBOS, dtype=np.int32)
_PAIR_J = np.zeros(N_COMBOS, dtype=np.int32)
for _idx, (_a, _b) in enumerate(
        (i, j) for i in range(N) for j in range(i + 1, N)):
    _PAIR_I[_idx] = _a
    _PAIR_J[_idx] = _b


def _seed_order():
    """Permutation of comparator indices forming the seed network (153 comps)."""
    return np.array([_pair_to_idx(p) for p in _SEED_PAIRS], dtype=np.int32)


def _apply(order):
    """Apply the comparator order to ALL 2^18 Boolean masks; return final state.

    Returns an (n_comps, 2^18) int8 array of the wire state after each step.
    """
    n = len(order)
    if n == 0:
        return np.empty((0, 1 << N), dtype=np.int8)
    # Start: every wire equals the corresponding bit of the mask.
    cur = np.arange(1 << N, dtype=np.int8).reshape(1, -1)
    for _ in range(N - 1):
        cur = np.concatenate([cur, cur], axis=0)
    # cur shape (N, 2^18); row w = bit w of the mask.
    out = np.empty((n, 1 << N), dtype=np.int8)
    for k, ci in enumerate(order):
        a = _PAIR_I[ci]
        b = _PAIR_J[ci]
        xa = cur[a]
        xb = cur[b]
        cur[a] = np.minimum(xa, xb)
        cur[b] = np.maximum(xa, xb)
        out[k] = cur
    return out


def _is_sorted(out):
    """True if the final state (last row of `out`) is sorted on every mask."""
    if out.shape[0] == 0:
        return True
    final = out[-1]
    for w in range(N - 1):
        if np.any(final[w] > final[w + 1]):
            return False
    return True


def _count_valid(out):
    """Number of comparators that change the output on at least one mask."""
    n = out.shape[0]
    if n == 0:
        return 0
    c = 0
    for k in range(n):
        if not np.array_equal(out[k], out[k - 1] if k > 0 else out[0]):
            c += 1
    return c


def _remove_redundant(order):
    """Drop zero-based-redundant comparators. Returns (new_order, count, valid)."""
    out = _apply(order)
    if not _is_sorted(out):
        return order, len(order), False
    n = out.shape[0]
    keep = np.ones(n, dtype=bool)
    for k in range(n):
        if k > 0 and np.array_equal(out[k], out[k - 1]):
            keep[k] = False
    if keep.all():
        return order, n, True
    new_order = order[keep]
    # Re-verify the reduced network is still a valid sorter.
    out2 = _apply(new_order)
    if _is_sorted(out2):
        return new_order, new_order.shape[0], True
    return order, n, True  # fallback: keep original (still valid)


def construct():
    deadline = time.time() + DEADLINE
    rng = np.random.default_rng(SEED)

    order = _seed_order()
    out = _apply(order)
    valid = _is_sorted(out)
    # Initial redundancy reduction.
    order, best_count, valid = _remove_redundant(order)
    if not valid:
        return list(map(tuple, zip(_PAIR_I[order], _PAIR_J[order])))

    # Local search over the comparator order (size minimisation).
    n = order.shape[0]
    # Precompute per-step "changed" flag to speed redundancy accounting.
    while time.time() < deadline:
        improved = False
        # One full pass over adjacent swaps.
        for k in range(n - 1):
            if time.time() >= deadline:
                break
            order[k], order[k + 1] = order[k + 1], order[k]
            out2 = _apply(order)
            if not _is_sorted(out2):
                order[k], order[k + 1] = order[k + 1], order[k]
                continue
            c2 = _count_valid(out2)
            if c2 <= best_count:
                # accept (keeps validity; may reduce size)
                if c2 < best_count:
                    improved = True
                    # reduce the order itself to the non-redundant subset
                    order, best_count, _ = _remove_redundant(order)
                    n = order.shape[0]
                    break  # restart scan from 0 after a reduction
                # else: same count, keep the swap (neutral) to explore
            else:
                order[k], order[k + 1] = order[k + 1], order[k]
        if time.time() >= deadline:
            break
        if not improved:
            # No size reduction in a full pass; do a few neutral shuffle rounds
            # to escape plateaus, then stop if still no improvement.
            for _ in range(3):
                if time.time() >= deadline:
                    break
                for k in range(n - 1):
                    order[k], order[k + 1] = order[k + 1], order[k]
                    out2 = _apply(order)
                    if not _is_sorted(out2):
                        order[k], order[k + 1] = order[k + 1], order[k]
                    else:
                        c2 = _count_valid(out2)
                        if c2 < best_count:
                            order, best_count, _ = _remove_redundant(order)
                            n = order.shape[0]
                            break
                        order[k], order[k + 1] = order[k + 1], order[k]
                if time.time() >= deadline:
                    break

    # Final safety: verify and, if invalid, fall back to the seed network.
    out_final = _apply(order)
    if not _is_sorted(out_final):
        order = _seed_order()
    order, _, _ = _remove_redundant(order)

    return list(map(tuple, zip(_PAIR_I[order], _PAIR_J[order])))
