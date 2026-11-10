# EVOLVE-BLOCK-START
import itertools
import time
import random
from math import comb
from collections import defaultdict

def _greedy_cover_bitset(v, k, t, rng, deadline):
    """
    Greedy set cover using bitset-accelerated marginal gain computation.
    Each candidate k-subset is precomputed as a big-int mask over all C(v,t) t-subsets.
    Each step picks the candidate with maximum (mask & remaining).bit_count().
    """
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {ts: i for i, ts in enumerate(all_t)}
    
    # Precompute masks for all C(v,k) candidate blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build mask for each block: a big int where bit i is set iff all_t[i] is subset of block
    block_masks = [0] * n_blocks
    for bi, blk in enumerate(all_blocks):
        blk_set = set(blk)
        mask = 0
        for ti, ts in enumerate(all_t):
            if all(p in blk_set for p in ts):
                mask |= (1 << ti)
        block_masks[bi] = mask
    
    remaining = (1 << n_t) - 1
    blocks = []
    used = [False] * n_blocks
    
    while remaining != 0:
        if time.time() > deadline:
            break
        
        best_gain = -1
        best_bi = -1
        
        for bi in range(n_blocks):
            if used[bi]:
                continue
            gain = (block_masks[bi] & remaining).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_bi = bi
        
        if best_bi < 0:
            break
        
        used[best_bi] = True
        blocks.append(list(all_blocks[best_bi]))
        remaining &= ~block_masks[best_bi]
    
    return blocks

def _greedy_cover(v, k, t, rng, deadline):
    """
    Greedy set cover with hardness-based seed selection (original approach, kept as fallback).
    """
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    uncovered = [True] * n_t
    n_uncovered = n_t
    
    point_to_ts = defaultdict(list)
    for i, ts in enumerate(all_t):
        for p in ts:
            point_to_ts[p].append(i)
            
    blocks = []
    
    while n_uncovered > 0:
        if time.time() > deadline:
            break
            
        best_seed = None
        best_score = float('inf')
        
        candidates = []
        if n_uncovered < 200:
            candidates = [i for i in range(n_t) if uncovered[i]]
        else:
            attempts = 0
            while len(candidates) < 100 and attempts < 500:
                idx = rng.randint(0, n_t - 1)
                if uncovered[idx] and idx not in candidates:
                    candidates.append(idx)
                attempts += 1
            if not candidates:
                for i in range(n_t):
                    if uncovered[i]:
                        candidates.append(i)
                        break

        for i in candidates:
            ts = all_t[i]
            neighbors = set()
            for p in ts:
                for j in point_to_ts[p]:
                    if j != i and uncovered[j]:
                        neighbors.add(j)
            
            score = len(neighbors)
            if score < best_score:
                best_score = score
                best_seed = i
            elif score == best_score and rng.random() < 0.2:
                best_seed = i
                
        if best_seed is None:
            break
            
        block = list(all_t[best_seed])
        
        while len(block) < k:
            if time.time() > deadline:
                break
            in_block = set(block)
            best_x = None
            best_gain = -1
            
            rests = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in in_block:
                    continue
                gain = 0
                for rest in rests:
                    ts_new = tuple(sorted(rest + (x,)))
                    idx = t_index.get(ts_new)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                unused = [x for x in range(v) if x not in in_block]
                if not unused:
                    break
                best_x = rng.choice(unused)
            
            block.append(best_x)
            
        block.sort()
        blocks.append(block)
        
        for ts in itertools.combinations(block, t):
            idx = t_index[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                n_uncovered -= 1
                
    return blocks

def _is_valid(blocks, v, k, t):
    total = comb(v, t)
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    return len(covered) == total

def _local_search(blocks, v, k, t, rng, deadline):
    """
    Local search to reduce block count.
    1. Remove redundant blocks.
    2. Replace a block with another that covers its unique t-subsets.
    """
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    blocks = [sorted(list(b)) for b in blocks]
    
    cov = [0] * n_t
    for b in blocks:
        for ts in itertools.combinations(b, t):
            cov[t_index[ts]] += 1
            
    def recompute_cov():
        nonlocal cov
        cov = [0] * n_t
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov[t_index[ts]] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        order = list(range(n))
        rng.shuffle(order)
        
        for bi in order:
            if time.time() > deadline:
                return blocks
            if bi >= len(blocks):
                continue
                
            b = blocks[bi]
            unique_idxs = []
            for ts in itertools.combinations(b, t):
                idx = t_index[ts]
                if cov[idx] == 1:
                    unique_idxs.append(idx)
                    
            if not unique_idxs:
                blocks.pop(bi)
                recompute_cov()
                improved = True
                break
                
            cand_set = set()
            for idx in unique_idxs:
                cand_set.update(all_t[idx])
            cand_list = list(cand_set)
            
            if len(cand_list) > k:
                continue
            
            # Try to find a replacement block
            # The new block must contain all points in cand_set (to cover unique t-subsets)
            # and fill remaining slots to maximize coverage
            
            base = list(cand_set)
            if len(base) < k:
                # Fill remaining slots greedily
                in_block = set(base)
                newblock = list(base)
                
                while len(newblock) < k:
                    best_x = None
                    best_gain = -1
                    rests = list(itertools.combinations(newblock, t - 1))
                    
                    for x in range(v):
                        if x in in_block:
                            continue
                        gain = 0
                        for rest in rests:
                            ts_new = tuple(sorted(rest + (x,)))
                            idx = t_index.get(ts_new)
                            if idx is not None:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None:
                        break
                    newblock.append(best_x)
                    in_block.add(best_x)
            else:
                newblock = base[:]
            
            newblock.sort()
            
            if newblock != b and newblock not in blocks:
                blocks.pop(bi)
                blocks.append(newblock)
                recompute_cov()
                improved = True
                break
                
    return blocks

def construct(v, k, t):
    start = time.time()
    deadline = start + 112.0
    best_blocks = None
    best_len = float('inf')
    
    # Try bitset greedy first - it should be much faster for C(24,7,3)
    rng = random.Random(42)
    b = _greedy_cover_bitset(v, k, t, rng, deadline - 20)
    if b and _is_valid(b, v, k, t):
        best_blocks = b
        best_len = len(b)
    
    if best_blocks is not None and time.time() < deadline - 10:
        ls = _local_search(best_blocks, v, k, t, rng, deadline - 10)
        if _is_valid(ls, v, k, t) and len(ls) < best_len:
            best_blocks = ls
            best_len = len(ls)
    
    # Try original greedy as well if time permits
    if time.time() < deadline - 20:
        rng2 = random.Random(12345)
        b2 = _greedy_cover(v, k, t, rng2, deadline - 20)
        if b2 and _is_valid(b2, v, k, t):
            if len(b2) < best_len:
                best_len = len(b2)
                best_blocks = b2
            if time.time() < deadline - 5:
                ls2 = _local_search(b2, v, k, t, rng2, deadline - 5)
                if _is_valid(ls2, v, k, t) and len(ls2) < best_len:
                    best_blocks = ls2
                    best_len = len(ls2)
    
    # More restarts with bitset greedy
    attempt = 1
    while time.time() < deadline - 10:
        rng = random.Random(1000 + attempt)
        b = _greedy_cover_bitset(v, k, t, rng, deadline - 10)
        if b and _is_valid(b, v, k, t):
            if len(b) < best_len:
                best_len = len(b)
                best_blocks = b
            if time.time() < deadline - 5:
                ls = _local_search(b, v, k, t, rng, deadline - 5)
                if _is_valid(ls, v, k, t) and len(ls) < best_len:
                    best_blocks = ls
                    best_len = len(ls)
        attempt += 1
    
    if best_blocks is None:
        return []
        
    return best_blocks
# EVOLVE-BLOCK-END
