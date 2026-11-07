# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _greedy_fallback():
    n = P['n']
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(a, b) in es for i, a in enumerate(trial) for b in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return [[int(v) for v in p] for p in parts]

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin under 90s kill

    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute validity for all 2^n subsets
    # valid[mask] = True if mask induces a clique or an independent set
    num_masks = 1 << n
    valid = [False] * num_masks
    valid[0] = True

    # For each mask, check if it's a clique or independent set
    # A set is a clique if for every pair (u,v) in it, u is in adj[v]
    # A set is an independent set if for every pair (u,v) in it, u is NOT in adj[v]
    for mask in range(1, num_masks):
        # Get the set of vertices
        verts = []
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            verts.append(v)
            m ^= lb

        # Check clique: every pair must be adjacent
        is_clique = True
        is_indep = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if not (adj[u] & (1 << v)):
                    is_clique = False
                else:
                    is_indep = False
                if not is_clique and not is_indep:
                    break
            if not is_clique and not is_indep:
                break

        valid[mask] = is_clique or is_indep

    # Build list of valid masks sorted by size descending
    valid_masks = [m for m in range(1, num_masks) if valid[m]]
    valid_masks.sort(key=lambda m: m.bit_count(), reverse=True)

    full_mask = (1 << n) - 1
    timed_out = [False]

    @lru_cache(maxsize=None)
    def min_parts(rem_mask):
        if rem_mask == 0:
            return 0
        if timed_out[0]:
            return 100
        if time.time() > deadline:
            timed_out[0] = True
            return 100
        best_add = 100
        for cm in valid_masks:
            if (cm & rem_mask) == cm:  # cm is subset of rem_mask
                r = rem_mask ^ cm
                add = 1 + min_parts(r)
                if add < best_add:
                    best_add = add
                    if best_add == 1:
                        break
        return best_add

    result = min_parts(full_mask)

    if timed_out[0] or result >= 100:
        parts = _greedy_fallback()
    else:
        # Reconstruct the partition
        def reconstruct(rem_mask):
            if rem_mask == 0:
                return []
            for cm in valid_masks:
                if (cm & rem_mask) == cm:
                    r = rem_mask ^ cm
                    if 1 + min_parts(r) == min_parts(rem_mask):
                        rest = reconstruct(r)
                        part = [v for v in range(n) if (cm >> v) & 1]
                        part.sort()
                        return [part] + rest
            return []

        parts = reconstruct(full_mask)

    # Verify
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))
    all_used = []
    valid_partition = True
    for part in parts:
        all_used.extend(part)
        adj_pairs = 0
        total_pairs = len(part) * (len(part) - 1) // 2
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                if (part[i], part[j]) in es:
                    adj_pairs += 1
        if not (adj_pairs == 0 or adj_pairs == total_pairs):
            valid_partition = False
            break

    if not valid_partition or len(all_used) != n or len(set(all_used)) != n:
        parts = _greedy_fallback()

    result = [list(sorted([int(v) for v in part])) for part in parts]
    return result

# EVOLVE-BLOCK-END
