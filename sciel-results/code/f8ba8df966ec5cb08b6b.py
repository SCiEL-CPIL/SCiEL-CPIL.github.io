# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_candidates():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # All maximal cliques
    cliques = []
    def gen_cliques(cand, cur):
        if not cand:
            if cur:
                cliques.append(frozenset(cur))
            return
        for i in range(len(cand)):
            v = cand[i]
            new_cand = [x for x in cand[i+1:] if v in adj[x]]
            gen_cliques(new_cand, cur + [v])
    gen_cliques(list(range(n)), [])

    # All maximal independent sets (maximal cliques of complement)
    non_adj = [set() for _ in range(n)]
    for i in range(n):
        non_adj[i] = set(range(n)) - adj[i] - {i}
    indep_sets = []
    def gen_is(cand, cur):
        if not cand:
            if cur:
                indep_sets.append(frozenset(cur))
            return
        for i in range(len(cand)):
            v = cand[i]
            new_cand = [x for x in cand[i+1:] if v in non_adj[x]]
            gen_is(new_cand, cur + [v])
    gen_is(list(range(n)), [])

    # Union of all maximal cliques and all maximal independent sets
    all_cands = list(set(cliques) | set(indep_sets))
    all_cands = [c for c in all_cands if len(c) >= 1]
    all_cands.sort(key=len, reverse=True)
    return all_cands

def _solve(candidates, n):
    full_mask = (1 << n) - 1
    # candidate masks
    cand_masks = [0] * len(candidates)
    for i, c in enumerate(candidates):
        m = 0
        for v in c:
            m |= (1 << v)
        cand_masks[i] = m

    @lru_cache(maxsize=None)
    def memo(rem_mask):
        """Returns minimum parts needed to cover rem_mask."""
        if rem_mask == 0:
            return 0
        best_add = 100
        for cm in cand_masks:
            if (cm & rem_mask) == cm:  # candidate is subset of remaining
                r = rem_mask ^ cm
                add = 1 + memo(r)
                if add < best_add:
                    best_add = add
        return best_add

    memo(full_mask)

    # Reconstruct
    def reconstruct(rem_mask):
        if rem_mask == 0:
            return []
        for i, cm in enumerate(cand_masks):
            if (cm & rem_mask) == cm:
                r = rem_mask ^ cm
                if 1 + memo(r) == memo(rem_mask):
                    part = [v for v in range(n) if (cm >> v) & 1]
                    part.sort()
                    return [part] + reconstruct(r)
        return []

    parts = reconstruct(full_mask)
    return parts

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    candidates = _build_candidates()
    parts = _solve(candidates, P['n'])

    # Verify
    n = P['n']
    all_used = []
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))
    valid = True
    for part in parts:
        all_used.extend(part)
        adj_pairs = 0
        total_pairs = len(part) * (len(part) - 1) // 2
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                if (part[i], part[j]) in es:
                    adj_pairs += 1
        if not (adj_pairs == 0 or adj_pairs == total_pairs):
            valid = False

    if len(all_used) != n or len(set(all_used)) != n or not valid:
        parts = _greedy_fallback()

    # Convert to list of lists of Python ints
    result = [list(sorted([int(v) for v in part])) for part in parts]
    return result

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
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return [[int(v) for v in p] for p in parts]

# EVOLVE-BLOCK-END
