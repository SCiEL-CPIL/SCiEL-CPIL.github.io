"""Randomized local search for covering design C(24,7,3) with block-deletion heuristic."""
import itertools
import random
import time
import math


def _block_tsubsets(block, t):
    """Return list of frozensets of all t-subsets of block."""
    return [frozenset(c) for c in itertools.combinations(block, t)]


def _scarcity_weighted_greedy(v, k, t, rng):
    """
    Dynamic scarcity-weighted greedy: prioritize uncovered triples whose points
    are 'rare' (appear in few existing blocks), reducing entropy of uncovered set.
    """
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    block_set = set()
    point_block_count = [0] * v  # how many blocks each point is in

    while uncovered:
        # Compute scarcity weight for each uncovered triple: sum of 1/(block_count+1) for its points
        # Pick the uncovered triple with highest scarcity weight (ties broken randomly)
        best_seed = None
        best_weight = -1
        # Sample a random subset of uncovered to avoid O(n) scan each time
        sample_size = min(200, len(uncovered))
        unc_list = list(uncovered)
        rng.shuffle(unc_list)
        sample = unc_list[:sample_size]
        
        for fs in sample:
            w = 0.0
            for pt in fs:
                w += 1.0 / (point_block_count[pt] + 1)
            if w > best_weight:
                best_weight = w
                best_seed = fs
        
        if best_seed is None:
            best_seed = rng.choice(unc_list)
        
        block = list(best_seed)
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
                # Add small scarcity bonus
                gain += 0.01 / (point_block_count[x] + 1)
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
            for pt in block:
                point_block_count[pt] += 1
            for fs in _block_tsubsets(block, t):
                uncovered.discard(fs)
    
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0

    all_tsubsets_list = [frozenset(c) for c in itertools.combinations(range(v), t)]
    total_t = len(all_tsubsets_list)

    # Build initial valid covering using scarcity-weighted greedy
    blocks = _scarcity_weighted_greedy(v, k, t, rng)
    
    # Pre-compute t-subsets for each block
    block_tsubs = [_block_tsubsets(b, t) for b in blocks]
    
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

    def try_remove_block(i):
        """Try to remove block i. Returns True if removed."""
        tsubs_i = block_tsubs[i]
        for fs in tsubs_i:
            if counts.get(fs, 0) <= 1:
                return False
        delta_u = apply_block_change(tsubs_i, -1)
        uncovered_val = uncovered + delta_u
        if uncovered_val == 0:
            blocks.pop(i)
            block_tsubs.pop(i)
            nonlocal uncovered
            uncovered = uncovered_val
            return True
        else:
            apply_block_change(tsubs_i, 1)
            return False

    # Block-Deletion Heuristic based on Criticality Analysis
    # Phase 1: Aggressive block deletion with repair
    def deletion_with_repair(max_rounds=50):
        """Iteratively remove blocks and repair gaps."""
        rounds = 0
        while rounds < max_rounds and time.time() - start_time < time_limit:
            rounds += 1
            # Find the block with the fewest critical triples
            best_i = -1
            best_critical = float('inf')
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            for i in indices:
                tsubs_i = block_tsubs[i]
                critical_count = 0
                for fs in tsubs_i:
                    if counts.get(fs, 0) == 1:
                        critical_count += 1
                        if critical_count >= best_critical:
                            break
                if critical_count < best_critical:
                    best_critical = critical_count
                    best_i = i
                if critical_count == 0:
                    break
            
            if best_i == -1:
                break
            
            # Try to remove block best_i
            old_blocks = list(blocks)
            old_block_tsubs = [list(ts) for ts in block_tsubs]
            old_counts = dict(counts)
            old_uncovered = uncovered
            
            tsubs_i = block_tsubs[best_i]
            delta_u = apply_block_change(tsubs_i, -1)
            uncovered_val = uncovered + delta_u
            blocks.pop(best_i)
            block_tsubs.pop(best_i)
            
            if uncovered_val == 0:
                # Successfully removed a block
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                continue
            
            # Need to repair: add blocks to cover the newly uncovered triples
            unc_list = []
            for c in all_tsubsets_list:
                if counts.get(c, 0) == 0:
                    unc_list.append(c)
            
            repair_success = False
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
                        for rest in itertools.combinations(new_b, t - 1):
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
                    new_tsubs = _block_tsubsets(new_b, t)
                    blocks.append(new_b_tuple)
                    block_tsubs.append(new_tsubs)
                    delta_u_add = apply_block_change(new_tsubs, 1)
                    uncovered_val += delta_u_add
                    
                    if uncovered_val == 0:
                        repair_success = True
                        break
                    
                    unc_list = []
                    for c in all_tsubsets_list:
                        if counts.get(c, 0) == 0:
                            unc_list.append(c)
                
                if time.time() - start_time > time_limit - 5:
                    break
            
            if repair_success and len(blocks) < len(old_blocks):
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
            else:
                # Revert
                blocks = old_blocks
                block_tsubs = old_block_tsubs
                counts.clear()
                counts.update(old_counts)
                uncovered = old_uncovered
    
    # Phase 2: Simulated annealing for swaps
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
                    for rest in itertools.combinations(new_block_set, t - 1):
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
                        new_tsubs = _block_tsubsets(list(new_block), t)
                        
                        delta_u_remove = apply_block_change(old_tsubs, -1)
                        delta_u_add = apply_block_change(new_tsubs, 1)
                        
                        new_uncovered = uncovered + delta_u_remove + delta_u_add
                        
                        old_energy = uncovered
                        new_energy = new_uncovered
                        delta_e = new_energy - old_energy
                        
                        accept = False
                        if delta_e <= 0:
                            accept = True
                        elif T > T_min:
                            if delta_e < 50:
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
        
        # 3. Periodically run deletion heuristic
        if step % 500 == 0 and len(blocks) > best_count + 2:
            deletion_with_repair(max_rounds=20)
        
        # 4. Perturbation if stuck
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
                            for rest in itertools.combinations(new_b, t - 1):
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
                        new_tsubs = _block_tsubsets(new_b, t)
                        blocks.append(new_b_tuple)
                        block_tsubs.append(new_tsubs)
                        delta_u_add = apply_block_change(new_tsubs, 1)
                        uncovered += delta_u_add
                        
                        unc_list = []
                        for c in all_tsubsets_list:
                            if counts.get(c, 0) == 0:
                                unc_list.append(c)
                    
                    if time.time() - start_time > time_limit - 5:
                        break
                
                if uncovered == 0:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    no_improve_iter = 0
                else:
                    blocks = old_blocks
                    block_tsubs = old_block_tsubs
                    counts.clear()
                    counts.update(old_counts)
                    uncovered = old_uncovered
                    no_improve_iter += 1
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1
        
        # Cool down
        T *= cooling_rate

    return [list(b) for b in best_blocks]
