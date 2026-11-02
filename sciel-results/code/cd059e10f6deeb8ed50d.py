"""Covering design C(24,7,3): bitset greedy + swap local search + deletion."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

_cache = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _cache:
        return _cache[key]
    triples = list(itertools.combinations(range(v), t))
    triple_index = {tr: i for i, tr in enumerate(triples)}
    ntriples = len(triples)
    t1subs = list(itertools.combinations(range(v), t - 1))
    ext = {}
    for sub in t1subs:
        d = {}
        for x in range(v):
            if x not in sub:
                tr = tuple(sorted(sub + (x,)))
                d[x] = triple_index[tr]
        ext[sub] = d
    triple_index_inv = {i: tr for tr, i in triple_index.items()}
    _cache[key] = (triples, triple_index, ntriples, t1subs, ext, triple_index_inv)
    return _cache[key]


def _block_bitset(block, triple_index, ntriples):
    bits = np.zeros(ntriples, dtype=np.int8)
    for tr in itertools.combinations(block, 3):
        bits[triple_index[tr]] = 1
    return bits


def _deletion_pass(blocks, block_bits, cov, deadline):
    removed = False
    i = 0
    while i < len(blocks):
        if time.time() > deadline - 5.0:
            break
        b = block_bits[i]
        if np.all(cov[b == 1] >= 2):
            cov = cov - b
            blocks.pop(i)
            block_bits.pop(i)
            removed = True
        else:
            i += 1
    return cov, removed


def _run_once(v, k, t, triples, triple_index, ntriples, t1subs, ext,
              triple_index_inv, seed, deadline):
    r = random.Random(seed)
    uncovered = np.ones(ntriples, dtype=np.int8)
    blocks = []
    block_bits = []
    while True:
        if time.time() > deadline - 5.0:
            return None
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        start = unc_idx[r.randint(0, len(unc_idx) - 1)]
        block = list(triple_index_inv[start])
        in_block = np.zeros(v, dtype=bool)
        for b in block:
            in_block[b] = True
        while len(block) < k:
            if time.time() > deadline - 5.0:
                return None
            block_t1 = list(itertools.combinations(block, t - 1))
            best_gain = -1
            best_x = -1
            for x in range(v):
                if in_block[x]:
                    continue
                gain = 0
                for sub in block_t1:
                    tr = ext[sub][x]
                    if uncovered[tr]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            in_block[best_x] = True
        block.sort()
        bits = _block_bitset(block, triple_index, ntriples)
        blocks.append(block)
        block_bits.append(bits)
        uncovered = np.logical_and(uncovered, 1 - bits)
    cov = np.sum(block_bits, axis=0) if block_bits else np.zeros(ntriples, dtype=np.int8)
    # deletion
    cov, _ = _deletion_pass(blocks, block_bits, cov, deadline)
    # swap local search to build redundancy then delete
    for it in range(60):
        if time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 5.0:
                break
            blk = blocks[i]
            oldbits = block_bits[i]
            for pos in range(k):
                old = blk[pos]
                for new in range(v):
                    if new == old or new in blk:
                        continue
                    newblk = blk[:pos] + [new] + blk[pos+1:]
                    newblk.sort()
                    nb = _block_bitset(newblk, triple_index, ntriples)
                    delta = nb - oldbits
                    gain = int(np.sum((cov == 1) & (delta == 1))) \
                         - int(np.sum((cov == 2) & (delta == -1)))
                    if gain > 0:
                        cov = cov + delta
                        blocks[i] = newblk
                        block_bits[i] = nb
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            cov, rem = _deletion_pass(blocks, block_bits, cov, deadline)
            if not rem:
                break
    cov, _ = _deletion_pass(blocks, block_bits, cov, deadline)
    return blocks


def construct(v, k, t):
    deadline = time.time() + 115.0
    triples, triple_index, ntriples, t1subs, ext, triple_index_inv = \
        _precompute(v, k, t)
    best_blocks = None
    best_count = 10**9
    seed = 0
    while time.time() < deadline - 3.0:
        res = _run_once(v, k, t, triples, triple_index, ntriples, t1subs,
                        ext, triple_index_inv, seed, deadline)
        seed += 1
        if res is None:
            continue
        if len(res) < best_count:
            best_count = len(res)
            best_blocks = res
        if best_count <= 70:
            break
    if best_blocks is None:
        # deterministic single greedy fallback
        res = _run_once(v, k, t, triples, triple_index, ntriples, t1subs,
                        ext, triple_index_inv, 0, deadline + 1000)
        best_blocks = res if res else []
    return best_blocks
# EVOLVE-BLOCK-END
