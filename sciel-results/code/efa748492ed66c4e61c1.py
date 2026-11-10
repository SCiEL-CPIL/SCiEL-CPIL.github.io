"""Cochromatic partition solver with branch-and-bound."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

# Module-level precomputation (done once)
_n = P['n']
_size = 1 << _n
_adj = [0] * _n
for _e in P['edges']:
    _a, _b = _e
    _adj[_a] |= (1 << _b)
    _adj[_b] |= (1 << _a)

_is_clique = [False] * _size
_is_ind = [False] * _size
_is_clique[0] = True
_is_ind[0] = True
for _s in range(1, _size):
    _lb = _s & (-_s)
    _v = _lb.bit_length() - 1
    _rest = _s ^ _lb
    if _is_clique[_rest] and (_adj[_v] & _rest) == _rest:
        _is_clique[_s] = True
    if _is_ind[_rest] and (_adj[_v] & _rest) == 0:
        _is_ind[_s] = True

_max_clique = 0
_max_ind = 0
for _s in range(1, _size):
    _c = bin(_s).count('1')
    if _is_clique[_s] and _c > _max_clique:
        _max_clique = _c
    if _is_ind[_s] and _c > _max_ind:
        _max_ind = _c

_lower_bound = max(math.ceil(_n / _max_ind), math.ceil(_n / _max_clique))


def _greedy_fallback():
    n = P['n']
    es = set()
    for e in P['edges']:
        es.add((e[0], e[1]))
        es.add((e[1], e[0]))
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            is_cl = True
            is_ind = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    if (part[i], part[j]) in es:
                        is_ind = False
                    else:
                        is_cl = False
            if not is_cl and not is_ind:
                continue
            new_cl = is_cl
            new_ind = is_ind
            for u in part:
                if (u, v) in es:
                    new_ind = False
                else:
                    new_cl = False
            if new_cl or new_ind:
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts


def _valid(mask):
    return _is_clique[mask] or _is_ind[mask]


def _can_partition(mask, parts_left, deadline):
    if mask == 0:
        return True
    if parts_left == 0:
        return False
    if time.time() > deadline:
        return False
    pop = bin(mask).count('1')
    if pop > parts_left * max(_max_clique, _max_ind):
        return False
    # Try all nonempty subsets of mask that are valid
    sub = mask
    while sub:
        if _valid(sub):
            if _can_partition(mask ^ sub, parts_left - 1, deadline):
                return True
        sub = (sub - 1) & mask
    return False


def _extract_partition(mask, parts_left, deadline):
    """Extract actual partition masks."""
    if mask == 0:
        return []
    if parts_left == 0:
        return None
    if time.time() > deadline:
        return None
    sub = mask
    while sub:
        if _valid(sub):
            rest = _extract_partition(mask ^ sub, parts_left - 1, deadline)
            if rest is not None:
                return [sub] + rest
        sub = (sub - 1) & mask
    return None


def construct():
    deadline = time.time() + 75.0
    n = P['n']
    full_mask = (1 << n) - 1

    for k in range(_lower_bound, n + 1):
        if time.time() > deadline:
            break
        if _can_partition(full_mask, k, deadline):
            parts_masks = _extract_partition(full_mask, k, deadline)
            if parts_masks is not None and len(parts_masks) == k:
                result = []
                for pm in parts_masks:
                    verts = [i for i in range(n) if pm & (1 << i)]
                    result.append(verts)
                # Verify
                seen = set()
                valid = True
                for part in result:
                    for v in part:
                        if v in seen:
                            valid = False
                            break
                        seen.add(v)
                    if not valid:
                        break
                    is_cl = True
                    is_ind = True
                    for i in range(len(part)):
                        for j in range(i + 1, len(part)):
                            if _adj[part[i]] & (1 << part[j]):
                                is_ind = False
                            else:
                                is_cl = False
                    if not (is_cl or is_ind):
                        valid = False
                        break
                if valid and len(seen) == n:
                    return result
            break
    return _greedy_fallback()
# EVOLVE-BLOCK-END
