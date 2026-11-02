"""Covering code K_q(n,r): numpy bitset greedy + local search with perturbation."""
import time
import itertools
import numpy as np


def _now():
    return time.time()


def _build_neighbor_masks(q, n, radius):
    """Return (total, masks) where masks[v] is a numpy bool array of length total,
    True for words within Hamming distance radius of word v."""
    total = q ** n
    # Precompute all words as tuples
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    masks = np.zeros((total, total), dtype=bool)
    for v in range(total):
        t = words[v]
        masks[v, v] = True
        if radius >= 1:
            for i in range(n):
                for a in range(q):
                    if a == t[i]:
                        continue
                    tt = list(t)
                    tt[i] = a
                    u = word_to_idx[tuple(tt)]
                    masks[v, u] = True
    return total, masks, words, word_to_idx


def _greedy_set_cover(total, masks, rng, deadline, margin=15.0):
    """Greedy set-cover using bitset masks. Returns list of codeword indices."""
    uncovered = np.ones(total, dtype=bool)
    code = []
    while uncovered.any():
        gains = masks @ uncovered.astype(np.int32)
        # Add small random perturbation to break ties
        gains = gains + rng.random(total) * 0.1
        best = int(np.argmax(gains))
        if gains[best] <= 0.1:  # Only the perturbation, no real gain
            break
        code.append(best)
        uncovered = uncovered & ~masks[best]
        if _now() > deadline - margin:
            break
    return code


def _local_search(code_idx, masks, total, sphere_list, deadline, margin=15.0):
    """Local search: repeatedly remove redundant codewords until fixed point.
    sphere_list[i] = list of word indices covered by codeword i."""
    code = list(set(code_idx))
    if not code:
        return code
    
    # Compute coverage counts
    counts = np.zeros(total, dtype=np.int32)
    for ci in code:
        counts[masks[ci]] += 1
    
    changed = True
    while changed and _now() < deadline - margin:
        changed = False
        new_code = []
        for ci in code:
            # Check if ci is redundant: all words it covers have count > 1
            covered_by_ci = masks[ci]
            if (counts[covered_by_ci] > 1).all():
                # ci is redundant, remove it
                counts[covered_by_ci] -= 1
                changed = True
            else:
                new_code.append(ci)
        code = new_code
    return code


def _perturbation_local_search(best_code, masks, total, sphere_list, rng, deadline, margin=15.0):
    """Perturbation: drop 1-2 random codewords, repair uncovered, local search."""
    best = list(best_code)
    best_len = len(best)
    
    iterations = 0
    while _now() < deadline - margin:
        iterations += 1
        code = list(best)
        k = len(code)
        if k < 2:
            break
        
        # Drop 1 or 2 random codewords
        if iterations % 3 == 0 and k >= 2:
            d1 = rng.integers(0, k)
            d2 = rng.integers(0, k)
            if d1 == d2:
                d2 = (d2 + 1) % k
            new_code = [c for j, c in enumerate(code) if j not in (d1, d2)]
        else:
            d1 = rng.integers(0, k)
            new_code = [c for j, c in enumerate(code) if j != d1]
        
        # Find uncovered words
        covered = np.zeros(total, dtype=bool)
        for ci in new_code:
            covered |= masks[ci]
        uncovered = np.where(~covered)[0]
        
        if len(uncovered) == 0:
            # No uncovered, just local search
            repaired = _local_search(new_code, masks, total, sphere_list, deadline, margin)
            if len(repaired) < best_len:
                best_len = len(repaired)
                best = repaired
            continue
        
        # Repair: greedily add codewords to cover uncovered
        uncov_set = set(uncovered.tolist())
        added = []
        while uncov_set:
            best_c = -1
            best_cnt = 0
            for c in range(total):
                cnt = np.sum(masks[c][list(uncov_set)]) if uncov_set else 0
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_cnt <= 0:
                break
            added.append(best_c)
            for w in np.where(masks[best_c])[0]:
                uncov_set.discard(int(w))
        
        combined = new_code + added
        repaired = _local_search(combined, masks, total, sphere_list, deadline, margin)
        if len(repaired) < best_len:
            best_len = len(repaired)
            best = repaired
    
    return best


def construct(q, n, radius):
    start = time.time()
    deadline = start + 135.0  # Safety margin under 150s limit
    margin = 15.0
    
    # Build neighbor masks
    total, masks, words, word_to_idx = _build_neighbor_masks(q, n, radius)
    
    # Build sphere list for faster local search
    sphere_list = [np.where(masks[i])[0].tolist() for i in range(total)]
    
    rng = np.random.default_rng(42)
    
    # Starting point: tetracode direct sum for ternary r=1, or greedy otherwise
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_idx = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = tuple(h + list(tail))
                seed_idx.append(word_to_idx[w])
        best_code = sorted(set(seed_idx))
    else:
        best_code = _greedy_set_cover(total, masks, rng, deadline, margin)
    
    # Initial local search
    best_code = _local_search(best_code, masks, total, sphere_list, deadline, margin)
    best_len = len(best_code)
    
    # Multiple greedy restarts with different random seeds
    for trial in range(10):
        if _now() > deadline - margin:
            break
        rng2 = np.random.default_rng(1000 + trial)
        code = _greedy_set_cover(total, masks, rng2, deadline, margin)
        if len(code) < best_len:
            code = _local_search(code, masks, total, sphere_list, deadline, margin)
            if len(code) < best_len:
                best_len = len(code)
                best_code = code
    
    # Perturbation-based local search
    if _now() < deadline - margin:
        best_code = _perturbation_local_search(best_code, masks, total, sphere_list, rng, deadline, margin)
        best_len = len(best_code)
    
    # Final redundancy pruning
    if _now() < deadline - margin:
        best_code = _local_search(best_code, masks, total, sphere_list, deadline, margin)
    
    # Convert to list of lists
    result = [list(words[v]) for v in best_code]
    
    # Verify coverage
    covered = np.zeros(total, dtype=bool)
    for v in best_code:
        covered |= masks[v]
    if not covered.all():
        # Fallback: use greedy to ensure coverage
        code = _greedy_set_cover(total, masks, np.random.default_rng(0), deadline, margin)
        result = [list(words[v]) for v in code]
    
    return result
