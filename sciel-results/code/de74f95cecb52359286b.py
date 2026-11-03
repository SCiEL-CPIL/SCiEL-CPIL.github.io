"""Covering design C(v,k,t) via Greedy + Iterated Local Search."""
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0
    rng = random.Random(20240517)
    
    triples = list(itertools.combinations(range(v), t))
    T = len(triples)
    tidx = {tr: i for i, tr in enumerate(triples)}
    
    def block_to_mask(block):
        m = np.zeros(T, dtype=bool)
        for tr in itertools.combinations(block, t):
            m[tidx[tr]] = True
        return m

    def is_valid(cov):
        return bool((cov >= 1).all())

    # Phase 1: Greedy Construction
    uncovered = np.ones(T, dtype=bool)
    blocks = []
    masks = []
    
    while uncovered.any():
        first = int(np.argmax(uncovered))
        tr0 = triples[first]
        block = list(tr0)
        
        while len(block) < k:
            best_x, best_gain = -1, -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                g = 0
                for r in itertools.combinations(block, t - 1):
                    tri = tuple(sorted(r + (x,)))
                    idx = tidx.get(tri)
                    if idx is not None and uncovered[idx]:
                        g += 1
                if g > best_gain:
                    best_gain, best_x = g, x
            block.append(best_x)
            
        block.sort()
        blocks.append(block)
        m = block_to_mask(block)
        masks.append(m)
        uncovered &= ~m

    best_blocks = list(blocks)
    best_count = len(best_blocks)
    
    # Phase 2: Iterated Local Search
    cur_blocks = list(blocks)
    cur_masks = list(masks)
    cur_cov = np.zeros(T, dtype=np.int32)
    for m in cur_masks:
        cur_cov += m.astype(np.int32)
        
    pool = []
    pool_masks = []
    def refresh_pool():
        nonlocal pool, pool_masks
        pool = []
        pool_masks = []
        # Add current blocks
        for b, m in zip(cur_blocks, cur_masks):
            pool.append(b)
            pool_masks.append(m)
        # Add random blocks
        for _ in range(1000):
            b = sorted(rng.sample(range(v), k))
            pool.append(b)
            pool_masks.append(block_to_mask(b))

    refresh_pool()
    
    last_improvement = time.time()
    
    while time.time() < deadline - 2.0:
        improved = False
        
        # 1. Removal Pass
        for i in range(len(cur_blocks)):
            if time.time() > deadline - 2.0:
                break
            counts_i = cur_cov[cur_masks[i]]
            if (counts_i >= 2).all():
                cur_cov -= cur_masks[i].astype(np.int32)
                del cur_blocks[i]
                del cur_masks[i]
                improved = True
                if len(cur_blocks) < best_count:
                    best_count = len(cur_blocks)
                    best_blocks = list(cur_blocks)
                    last_improvement = time.time()
                i = -1

        # 2. Swap Pass
        if time.time() > deadline - 2.0:
            break
            
        for _ in range(500):
            if time.time() > deadline - 2.0:
                break
            if not cur_blocks:
                break
            i = rng.randrange(len(cur_blocks))
            j = rng.randrange(len(pool))
            
            if cur_blocks[i] == pool[j]:
                continue
                
            mask_i = cur_masks[i]
            mask_j = pool_masks[j]
            
            critical_mask = mask_i & ~mask_j
            if critical_mask.any():
                counts_crit = cur_cov[critical_mask]
                if (counts_crit >= 2).all():
                    # Valid swap
                    cur_cov -= mask_i.astype(np.int32)
                    cur_cov += mask_j.astype(np.int32)
                    cur_blocks[i] = pool[j]
                    cur_masks[i] = mask_j
                    improved = True
                    if len(cur_blocks) < best_count:
                        best_count = len(cur_blocks)
                        best_blocks = list(cur_blocks)
                        last_improvement = time.time()
                    
                    # Quick removal check
                    for r in range(len(cur_blocks)):
                        counts_r = cur_cov[cur_masks[r]]
                        if (counts_r >= 2).all():
                            cur_cov -= cur_masks[r].astype(np.int32)
                            del cur_blocks[r]
                            del cur_masks[r]
                            improved = True
                            if len(cur_blocks) < best_count:
                                best_count = len(cur_blocks)
                                best_blocks = list(cur_blocks)
                                last_improvement = time.time()
                            break

        # 3. Perturbation if stuck
        if time.time() - last_improvement > 5.0:
            # Perturb: replace 5 random blocks
            num_perturb = min(5, len(cur_blocks))
            for _ in range(num_perturb):
                if not cur_blocks: break
                idx = rng.randrange(len(cur_blocks))
                new_b = sorted(rng.sample(range(v), k))
                new_m = block_to_mask(new_b)
                
                # This might make the covering invalid
                cur_cov -= cur_masks[idx].astype(np.int32)
                cur_cov += new_m.astype(np.int32)
                cur_blocks[idx] = new_b
                cur_masks[idx] = new_m
                
            # Repair: add blocks to cover uncovered triples
            while not is_valid(cur_cov):
                if time.time() > deadline - 2.0:
                    break
                # Find uncovered
                uncovered = cur_cov == 0
                if not uncovered.any():
                    break
                first = int(np.argmax(uncovered))
                tr0 = triples[first]
                # Find a block in pool that covers tr0? Or just random?
                # Random is safer to find a valid block
                b = sorted(rng.sample(range(v), k))
                m = block_to_mask(b)
                if m[tidx[tr0]]: # Must cover the specific triple
                    cur_cov += m.astype(np.int32)
                    cur_blocks.append(b)
                    cur_masks.append(m)
                else:
                    # If random block doesn't cover it, try a few more
                    for _ in range(10):
                        b = sorted(rng.sample(range(v), k))
                        m = block_to_mask(b)
                        if m[tidx[tr0]]:
                            cur_cov += m.astype(np.int32)
                            cur_blocks.append(b)
                            cur_masks.append(m)
                            break
                    else:
                        # Force add a block containing tr0
                        b = list(tr0)
                        while len(b) < k:
                            b.append(rng.randint(0, v-1))
                        # Ensure distinct
                        b_set = set(b)
                        while len(b_set) < k:
                            b_set.add(rng.randint(0, v-1))
                        b = sorted(b_set)
                        m = block_to_mask(b)
                        cur_cov += m.astype(np.int32)
                        cur_blocks.append(b)
                        cur_masks.append(m)
                        
            refresh_pool()
            last_improvement = time.time() # Reset timer to allow search

    # Final cleanup
    best_cov = np.zeros(T, dtype=np.int32)
    for b in best_blocks:
        best_cov += block_to_mask(b).astype(np.int32)
        
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for i in range(len(best_blocks)):
            m = block_to_mask(best_blocks[i])
            counts_i = best_cov[m]
            if (counts_i >= 2).all():
                best_cov -= m.astype(np.int32)
                del best_blocks[i]
                changed = True
                break

    return best_blocks
