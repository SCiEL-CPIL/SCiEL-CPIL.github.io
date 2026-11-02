# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

# Precomputed constants for the target parameters (v=19, k=7, t=4).
_V = 19
_K = 7
_T = 4
_T1 = _T - 1

# All t-subsets and (t-1)-subsets as tuples, plus a fast integer key mapping.
_ALL_T = list(itertools.combinations(range(_V), _T))
_ALL_T1 = list(itertools.combinations(range(_V), _T1))
_TKEY = {c: i for i, c in enumerate(_ALL_T)}
_T1KEY = {c: i for i, c in enumerate(_ALL_T1)}
_N_T = len(_ALL_T)
_N_T1 = len(_ALL_T1)

# Map block index -> block tuple (same order as itertools.combinations(range(_V),_K)).
_BLOCK_LIST = [list(c) for c in itertools.combinations(range(_V), _K)]
_NBLOCK = len(_BLOCK_LIST)

# Precompute the (t-1)-subset keys that each 7-block covers.
_BLOCK_T1 = []
for _b in _BLOCK_LIST:
    _BLOCK_T1.append(np.array([_T1KEY[c] for c in itertools.combinations(_b, _T1)],
                              dtype=np.int32))
_BLOCK_T1 = np.stack(_BLOCK_T1, axis=0)          # (nblocks, C(k, t-1))

# For each t-subset, the candidate blocks that cover it, each with a boolean
# mask over the distinct t-subsets that block covers.
_CAND = [None] * _N_T
for _bi in range(_NBLOCK):
    _keys = _BLOCK_T1[_bi]
    for _k in _keys:
        _k = int(_k)
        _lst = _CAND[_k]
        if _lst is None:
            _mask = np.zeros(_N_T, dtype=bool)
            _mask[_keys] = True
            _CAND[_k] = [(_bi, _mask)]
        else:
            _mask = np.zeros(_N_T, dtype=bool)
            _mask[_keys] = True
            _lst.append((_bi, _mask))


def _greedy(seed, deadline):
    """Vectorized greedy covering with a deterministic tie-break by seed noise."""
    rng = random.Random(seed)
    uncovered = np.ones(_N_T, dtype=bool)
    rem = np.full(_N_T, 1, dtype=np.int32)
    noise = np.random.RandomState(seed).rand(_V).astype(np.float64)
    blocks = []
    while uncovered.any():
        if time.time() >= deadline:
            break
        ti = int(np.argmax(uncovered))
        block = list(_ALL_T[ti])
        while len(block) < _K:
            if time.time() >= deadline:
                break
            subkeys = np.array([_T1KEY[c] for c in itertools.combinations(block, _T1)],
                               dtype=np.int32)
            best = -1
            best_score = -1.0
            for x in range(_V):
                if x in block:
                    continue
                cnt = int(rem[subkeys].sum())
                score = cnt + 1e-6 * noise[x]
                if score > best_score:
                    best_score = score
                    best = x
            if best < 0:
                break
            block.append(best)
        block.sort()
        if len(block) < _K:
            break
        bi = _TKEY[tuple(block)]
        keys = _BLOCK_T1[bi]
        uncovered[keys] = False
        rem[keys] -= 1
        blocks.append(block)
    return blocks


def _validate(blocks):
    cov = np.zeros(_N_T, dtype=bool)
    for b in blocks:
        cov[_BLOCK_T1[_TKEY[tuple(b)]]] = True
    return bool(cov.all())


def _delete(blocks, deadline):
    if not blocks:
        return blocks
    cnt = np.zeros(_N_T, dtype=np.int32)
    for b in blocks:
        cnt[_BLOCK_T1[_TKEY[tuple(b)]]] += 1
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() >= deadline:
                return blocks
            bi = _TKEY[tuple(blocks[i])]
            keys = _BLOCK_T1[bi]
            if cnt[keys].min() == 1:
                continue
            cnt[keys] -= 1
            blocks.pop(i)
            changed = True
    return blocks


def _local_search(blocks, seed, deadline):
    """WalkSAT-style stochastic local search over block selection."""
    rng = random.Random(seed)
    best = None
    best_len = len(blocks)

    def _counts(cur):
        c = np.zeros(_N_T, dtype=np.int32)
        for b in cur:
            c[_BLOCK_T1[_TKEY[tuple(b)]]] += 1
        return c

    def _cov(cur):
        c = np.zeros(_N_T, dtype=bool)
        for b in cur:
            c[_BLOCK_T1[_TKEY[tuple(b)]]] = True
        return c

    stall = 0
    for restart in range(200000):
        if time.time() >= deadline:
            break
        if restart == 0:
            cur = list(blocks)
        else:
            cur = list(blocks)
            for _ in range(len(cur) // 4):
                cur[rng.randrange(len(cur))] = None
            cur = [b for b in cur if b is not None]

        chosen = set(_TKEY[tuple(b)] for b in cur)
        cov = _cov(cur)
        counts = _counts(cur)
        uncovered = ~cov
        cur_len = len(cur)

        if not uncovered.any() and cur_len < best_len:
            best = list(cur)
            best_len = cur_len
            stall = 0
            continue

        stall += 1
        if stall > 400:
            continue

        for _ in range(60):
            if time.time() >= deadline:
                break
            if not uncovered.any():
                break
            ui = int(rng.choice(np.flatnonzero(uncovered)))
            cands = _CAND[ui]
            scored = []
            for bi, mask in cands:
                if bi in chosen:
                    continue
                newcov = int(mask[uncovered].sum())
                scored.append((newcov, bi, mask))
            if not scored:
                for bi, mask in cands:
                    newcov = int(mask[uncovered].sum())
                    scored.append((newcov, bi, mask))
            if not scored:
                break
            scored.sort(key=lambda z: (-z[0], z[1]))
            top = scored[:min(8, len(scored))]
            newcov, bi, mask = top[rng.randrange(len(top))]

            # choose the block to drop: the one with the fewest unique covers
            best_drop = None
            best_loss = None
            for b in cur:
                bbi = _TKEY[tuple(b)]
                bmask = _BLOCK_T1[bbi]
                loss = int((counts[bmask] == 1).sum())
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_drop = b
            accept = (newcov >= best_loss) or (rng.random() < 0.15 and newcov >= best_loss - 2)
            if not accept:
                continue
            # apply swap
            drop = best_drop
            dbi = _TKEY[tuple(drop)]
            counts[_BLOCK_T1[dbi]] -= 1
            cur.remove(drop)
            chosen.discard(dbi)
            cur.append(_BLOCK_LIST[bi])
            chosen.add(bi)
            counts[_BLOCK_T1[bi]] += 1
            cov[_BLOCK_T1[bi]] = True
            uncovered[_BLOCK_T1[bi]] = False
            cur_len = len(cur)
            if not uncovered.any() and cur_len < best_len:
                best = list(cur)
                best_len = cur_len
                stall = 0
                break

    return best if best is not None else blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0   # safety margin before the 150 s wall-clock kill
    if (v, k, t) != (_V, _K, _T):
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    best = None
    best_len = 10 ** 9
    seeds = list(range(64))
    rng = random.Random(0)
    for it in range(100000):
        if time.time() >= deadline:
            break
        seed = seeds[it % len(seeds)] if it < len(seeds) else rng.randrange(10 ** 9)
        blocks = _greedy(seed, deadline)
        if not blocks:
            continue
        blocks = _delete(blocks, deadline)
        if not _validate(blocks):
            continue
        if len(blocks) < best_len:
            best = blocks
            best_len = len(blocks)
        if time.time() < deadline - 20.0:
            ls = _local_search(blocks, seed, deadline - 10.0)
            if _validate(ls) and len(ls) < best_len:
                best = ls
                best_len = len(ls)
    if best is None:
        best = _greedy(0, time.time() + 1.0)
        best = _delete(best, time.time() + 1.0)
    return best
# EVOLVE-BLOCK-END
