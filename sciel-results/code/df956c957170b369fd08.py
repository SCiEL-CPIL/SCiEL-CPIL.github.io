"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    # Deadline guard: 150s wall-clock limit, keep 10s margin
    start_time = time.time()
    deadline = start_time + 140.0

    total_words = q ** n

    # If the space is small enough for a fast greedy, use it.
    if total_words <= 100000:
        if radius == 1:
            result = _fast_greedy_q1(q, n, deadline)
            if result is not None:
                return result
        else:
            if total_words <= 10000:
                result = _greedy_generic(q, n, radius, deadline)
                if result is not None:
                    return result

    # --- Fallback: Direct sum construction ---
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_n = 4
        base_code = tetracode
    elif q >= 3 and n >= 4 and radius == 1:
        base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        base_n = 4
    else:
        base_n = min(n, 4)
        base_code = []
        for tail in itertools.product(range(q), repeat=base_n):
            base_code.append(list(tail))

    if n < base_n:
        return [list(range(n))]

    tails = list(itertools.product(range(q), repeat=n - base_n))
    return [h + list(tail) for h in base_code for tail in tails]


def _fast_greedy_q1(q, n, deadline):
    """Optimized greedy + local search for q-ary, radius=1 covering."""
    total_words = q ** n
    sphere_size = 1 + n * (q - 1)

    # Build neighbor table
    neighbors = np.empty((total_words, sphere_size), dtype=np.int32)
    for i in range(total_words):
        nbrs = [i]
        for pos in range(n):
            pow_q = q ** (n - 1 - pos)
            current_val = (i // pow_q) % q
            for new_val in range(q):
                if new_val != current_val:
                    nbrs.append(i + (new_val - current_val) * pow_q)
        neighbors[i] = nbrs

    # Greedy set cover
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []

    while time.time() < deadline:
        gains = (~covered)[neighbors].sum(axis=1)
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        code_indices.append(best_idx)
        covered[neighbors[best_idx]] = True

    # Convert to list
    code_indices = list(code_indices)

    # --- Local search: removal pass ---
    # Try removing each codeword if coverage is maintained
    for attempt in range(len(code_indices)):
        if time.time() > deadline - 5:
            break
        changed = False
        for i in range(len(code_indices)):
            if time.time() > deadline - 5:
                break
            # Tentatively remove
            new_code = code_indices[:i] + code_indices[i+1:]
            # Check if still covered
            cov = np.zeros(total_words, dtype=bool)
            for ci in new_code:
                cov[neighbors[ci]] = True
            if cov.all():
                code_indices = new_code
                changed = True
                break
        if not changed:
            break

    # --- Local search: swap pass ---
    # Try swapping one codeword with another to reduce count or maintain
    for swap_round in range(3):
        if time.time() > deadline - 3:
            break
        improved = False
        for i in range(len(code_indices)):
            if time.time() > deadline - 3:
                break
            for j in range(len(code_indices)):
                if i == j:
                    continue
                if time.time() > deadline - 3:
                    break
                # Try removing i and adding j (if j not already present)
                if j in code_indices:
                    continue
                new_code = code_indices[:i] + code_indices[i+1:] + [j]
                cov = np.zeros(total_words, dtype=bool)
                for ci in new_code:
                    cov[neighbors[ci]] = True
                if cov.all():
                    code_indices = new_code
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Final removal pass after swaps
    for attempt in range(len(code_indices)):
        if time.time() > deadline - 2:
            break
        changed = False
        for i in range(len(code_indices)):
            if time.time() > deadline - 2:
                break
            new_code = code_indices[:i] + code_indices[i+1:]
            cov = np.zeros(total_words, dtype=bool)
            for ci in new_code:
                cov[neighbors[ci]] = True
            if cov.all():
                code_indices = new_code
                changed = True
                break
        if not changed:
            break

    # Convert indices to codewords
    result = []
    for idx in code_indices:
        w = []
        temp = idx
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    # Verify coverage (safety check)
    if _verify_coverage(result, q, n, 1):
        return result
    return None


def _verify_coverage(codewords, q, n, radius):
    """Check that every word is within Hamming distance radius of some codeword."""
    total_words = q ** n
    if radius == 1:
        covered = np.zeros(total_words, dtype=bool)
        for cw in codewords:
            idx = 0
            for i, v in enumerate(cw):
                idx += v * (q ** (n - 1 - i))
            covered[idx] = True
            for pos in range(n):
                pow_q = q ** (n - 1 - pos)
                cur = (idx // pow_q) % q
                for nv in range(q):
                    if nv != cur:
                        covered[idx + (nv - cur) * pow_q] = True
        return covered.all()
    else:
        # Generic: generate all words within distance radius
        covered = set()
        for cw in codewords:
            idx = 0
            for i, v in enumerate(cw):
                idx += v * (q ** (n - 1 - i))
            for delta in _all_deltas(n, q, radius):
                covered.add(idx + delta)
        return len(covered) == total_words


def _all_deltas(n, q, radius):
    """Yield index deltas for all words within Hamming distance radius of the zero word."""
    # For each subset of positions of size <= radius, and each non-zero change
    deltas = set()
    deltas.add(0)
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            # For each position, choose a non-zero delta value
            base_options = []
            for pos in positions:
                pow_q = q ** (n - 1 - pos)
                base_options.append([d * pow_q for d in range(1, q)])
            for combo in itertools.product(*base_options):
                deltas.add(sum(combo))
    return deltas


def _greedy_generic(q, n, radius, deadline):
    """Generic greedy for small spaces and any radius."""
    total_words = q ** n
    # Precompute neighbors
    neighbors_list = []
    for i in range(total_words):
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        nbrs = _neighbors_of_word(w, q, n, radius)
        # Convert back to indices
        idx_nbrs = []
        for nb in nbrs:
            idx = 0
            for j, v in enumerate(nb):
                idx += v * (q ** (n - 1 - j))
            idx_nbrs.append(idx)
        neighbors_list.append(idx_nbrs)

    covered = [False] * total_words
    code_indices = []

    while time.time() < deadline:
        best_idx = -1
        best_gain = 0
        for i in range(total_words):
            gain = sum(1 for nb in neighbors_list[i] if not covered[nb])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        code_indices.append(best_idx)
        for nb in neighbors_list[best_idx]:
            covered[nb] = True

    result = []
    for idx in code_indices:
        w = []
        temp = idx
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    return result


def _neighbors_of_word(word, q, n, radius):
    """Return all words within Hamming distance <= radius of the given word."""
    result = [list(word)]
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            base = list(word)
            for combo in itertools.product(*[range(q)] * r):
                for pos, val in zip(positions, combo):
                    if val != base[pos]:
                        base[pos] = val
                # Only include if different from original
                if base != list(word):
                    result.append(list(base))
            # Reset base for next combination of positions
            base = list(word)
    return result
# EVOLVE-BLOCK-END
