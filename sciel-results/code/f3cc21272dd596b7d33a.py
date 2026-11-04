import time
import itertools
import numpy as np

# Precompute the tetracode (size 9, length 4, radius 1)
_TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

def _tetracode_solution(q, n, radius):
    """Direct sum of the ternary tetracode with every word on the remaining coords."""
    if n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    return [h + list(tail) for h in _TETRA
            for tail in itertools.product(range(q), repeat=n - 4)]

def _encode(word, q, n):
    val = 0
    for i in range(n):
        val = val * q + word[i]
    return val

def _decode(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word

def _build_cover_counts(codewords, q, n):
    """
    Returns:
    cover_count: array of size q^n, cover_count[i] = number of codewords covering word i.
    """
    total = q ** n
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        idx = _encode(cw, q, n)
        cover_count[idx] += 1
        for i in range(n):
            v = cw[i]
            for d in range(q):
                if d == v:
                    continue
                # Neighbors
                nb = cw[:]
                nb[i] = d
                idx = _encode(nb, q, n)
                cover_count[idx] += 1
    return cover_count

def _find_redundant(codewords, cover_count, q, n):
    """
    Find codewords that can be removed without violating the cover property.
    """
    redundant = []
    for i, cw in enumerate(codewords):
        idx = _encode(cw, q, n)
        if cover_count[idx] == 1:
            continue
        redundant_candidate = True
        for i_pos in range(n):
            v = cw[i_pos]
            for d in range(q):
                if d == v:
                    continue
                nb = cw[:]
                nb[i_pos] = d
                idx = _encode(nb, q, n)
                if cover_count[idx] == 1:
                    redundant_candidate = False
                    break
            if not redundant_candidate:
                break
        if redundant_candidate:
            redundant.append(i)
    return redundant

def _remove_codeword(codewords, cover_count, idx, q, n):
    cw = codewords.pop(idx)
    cover_count[_encode(cw, q, n)] -= 1
    for i_pos in range(n):
        v = cw[i_pos]
        for d in range(q):
            if d == v:
                continue
            nb = cw[:]
            nb[i_pos] = d
            cover_count[_encode(nb, q, n)] -= 1

def _local_search(codewords, q, n, deadline):
    total = q ** n
    cover_count = _build_cover_counts(codewords, q, n)
    
    if not np.all(cover_count > 0):
        return codewords

    # Initial pruning
    redundant_indices = _find_redundant(codewords, cover_count, q, n)
    for i in sorted(redundant_indices, reverse=True):
        _remove_codeword(codewords, cover_count, i, q, n)
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to find a move that allows a removal
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            cw = codewords[i]
            # Try moving to neighbors
            for i_pos in range(n):
                v = cw[i_pos]
                for d in range(q):
                    if d == v:
                        continue
                    cw_new = cw[:]
                    cw_new[i_pos] = d
                    
                    # Simulate move: remove i, add cw_new
                    # Check validity
                    new_cover_count = cover_count.copy()
                    # Subtract cw
                    new_cover_count[_encode(cw, q, n)] -= 1
                    for pos in range(n):
                        vv = cw[pos]
                        for dd in range(q):
                            if dd == vv:
                                continue
                            nb = cw[:]
                            nb[pos] = dd
                            new_cover_count[_encode(nb, q, n)] -= 1
                            
                    # Add cw_new
                    new_cover_count[_encode(cw_new, q, n)] += 1
                    for pos in range(n):
                        vv = cw_new[pos]
                        for dd in range(q):
                            if dd == vv:
                                continue
                            nb = cw_new[:]
                            nb[pos] = dd
                            new_cover_count[_encode(nb, q, n)] += 1
                            
                    if not np.all(new_cover_count > 0):
                        continue
                        
                    # Check if any codeword becomes redundant
                    new_codewords = codewords[:i] + codewords[i+1:] + [cw_new]
                    redundant_found = -1
                    for j, c_j in enumerate(new_codewords):
                        idx = _encode(c_j, q, n)
                        if new_cover_count[idx] == 1:
                            continue
                        is_red = True
                        for pos in range(n):
                            vv = c_j[pos]
                            for dd in range(q):
                                if dd == vv:
                                    continue
                                nb = c_j[:]
                                nb[pos] = dd
                                if new_cover_count[_encode(nb, q, n)] == 1:
                                    is_red = False
                                    break
                            if not is_red:
                                break
                        if is_red:
                            redundant_found = j
                            break
                            
                    if redundant_found != -1:
                        codewords = new_codewords
                        cover_count = new_cover_count
                        _remove_codeword(codewords, cover_count, redundant_found, q, n)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
    return codewords

def construct(q, n, radius):
    deadline = time.time() + 140.0
    try:
        codewords = _tetracode_solution(q, n, radius)
        codewords = _local_search(codewords, q, n, deadline)
        return codewords
    except Exception:
        return _tetracode_solution(q, n, radius)
