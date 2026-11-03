"""Randomized local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math


def _precompute_block_tsubsets(block, t):
    """Pre-compute all t-subsets of a block as frozensets."""
    return [frozenset(c) for c in itertools.combinations(block, t)]


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    block_set = set()
    
    while uncovered:
        seed = rng.choice(list(uncovered))
        block = list(seed)
        rng.shuffle(block)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            for fs in _precompute_block_tsubsets(block, t):
                uncovered.discard(fs)
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0

    # Pre-compute all t-subsets for quick lookup
    all_tsubsets_list = [frozenset(c) for c in itertools.combinations(range(v), t)]
    total_t = len(all_tsubsets_list)

    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    
    # Pre-compute t-subsets for each block
    block_tsubs = [_precompute_block_tsubsets(b, t) for b in blocks]
    
    def compute_counts(blk_list, blk_tsubs_list):
        counts = {}
        for tsubs in blk_tsubs_list:
            for fs in tsubs:
                counts[fs] = counts.get(fs, 0) + 1
        return counts

    counts = compute_counts(blocks, block_tsubs)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    uncovered = total_t - len(counts)
    
    def apply_block_change(tsubs, sign):
        """Apply adding/removing a block's t-subsets to counts. Returns delta_uncovered."""
        delta_uncovered = 0
        for fs in tsubs:
            old_c = counts.get(fs, 0)
            new_c = old_c + sign
            if new_c == 0:
                if fs in counts:
                    del counts[fs]
                if old_c == 1:
                    delta_uncovered += 1
            elif new_c > 0:
                if old_c == 0:
                    delta_uncovered -= 1
                counts[fs] = new_c
        return delta_uncovered

    # Simulated annealing parameters
    T0 = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    T = T0
    
    no_improve_iter = 0
    max_no_improve = 300
    step = 0
    
    while time.time() - start_time < time_limit:
        step += 1
        
        # 1. Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            tsubs_i = block_tsubs[i]
            redundant = True
            for fs in tsubs_i:
                if counts.get(fs, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                delta_u = apply_block_change(tsubs_i, -1)
                uncovered += delta_u
                if uncovered == 0:
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed_any = True
                    no_improve_iter = 0
                    break
                else:
                    apply_block_change(tsubs_i, 1)
                    uncovered -= delta_u
        
        if removed_any:
            continue
        
        # 2. Try swaps with simulated annealing acceptance
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_tsubs = block_tsubs[i]
            
            critical = []
            for fs in old_tsubs:
                if counts.get(fs, 0) == 1:
                    critical.append(fs)
            
            if not critical:
                continue
            
            points_needed = set()
            for fs in critical:
                points_needed.update(fs)
            
            if len(points_needed) <= k:
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t-1):
                        fs = frozenset(list(rest) + [x])
                        c = counts.get(fs, 0)
                        if c == 0:
                            gain += 10
                        elif c == 1:
                            gain += 1
                    new_block_set.add(x)
                        
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    if new_block != old_block:
                        new_tsubs = _precompute_block_tsubsets(list(new_block), t)
                        
                        delta_u_remove = apply_block_change(old_tsubs, -1)
                        delta_u_add = apply_block_change(new_tsubs, 1)
                        
                        new_uncovered = uncovered + delta_u_remove + delta_u_add
                        
                        # Energy: uncovered count (lower is better)
                        old_energy = uncovered
                        new_energy = new_uncovered
                        delta_e = new_energy - old_energy
                        
                        accept = False
                        if delta_e <= 0:
                            accept = True
                        elif T > T_min:
                            if delta_e < 50:  # only accept small increases
                                prob = math.exp(-delta_e / T)
                                accept = (rng.random() < prob)
                        
                        if accept:
                            blocks[i] = new_block
                            block_tsubs[i] = new_tsubs
                            uncovered = new_uncovered
                            no_improve_iter = 0
                        else:
                            apply_block_change(new_tsubs, -1)
                            apply_block_change(old_tsubs, 1)
                            uncovered = old_energy
        
        # 3. Perturbation if stuck
        if no_improve_iter > max_no_improve:
            if len(blocks) > 1:
                old_blocks = list(blocks)
                old_block_tsubs = [list(ts) for ts in block_tsubs]
                old_counts = dict(counts)
                old_uncovered = uncovered
                
                i = rng.randrange(len(blocks))
                b = blocks[i]
                tsubs_i = block_tsubs[i]
                
                delta_u = apply_block_change(tsubs_i, -1)
                uncovered += delta_u
                blocks.pop(i)
                block_tsubs.pop(i)
                
                unc_list = []
                for c in all_tsubsets_list:
                    if counts.get(c, 0) == 0:
                        unc_list.append(c)
                
                while unc_list:
                    seed = rng.choice(unc_list)
                    new_b = list(seed)
                    rng.shuffle(new_b)
                    
                    while len(new_b) < k:
                        best_x = None
                        best_gain = -1
                        cands = [x for x in range(v) if x not in new_b]
                        rng.shuffle(cands)
                        for x in cands:
                            gain = 0
                            for rest in itertools.combinations(new_b, t-1):
                                fs = frozenset(list(rest) + [x])
                                if counts.get(fs, 0) == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        new_b.append(best_x)
                    
                    new_b.sort()
                    new_b_tuple = tuple(new_b)
                    
                    if new_b_tuple not in set(blocks):
                        new_tsubs = _precompute_block_tsubsets(new_b, t)
                        blocks.append(new_b_tuple)
                        block_tsubs.append(new_tsubs)
                        delta_u_add = apply_block_change(new_tsubs, 1)
                        uncovered += delta_u_add
                        
                        unc_list = []
                        for c in all_tsubsets_list:
                            if counts.get(c, 0) == 0:
                                unc_list.append(c)
                    
                if uncovered == 0:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    no_improve_iter = 0
                else:
                    blocks = old_blocks
                    block_tsubs = old_block_tsubs
                    counts = old_counts
                    uncovered = old_uncovered
                    no_improve_iter += 1
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1
        
        # Cool down
        T *= cooling_rate

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
