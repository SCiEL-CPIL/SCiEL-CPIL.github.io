"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math


def _greedy_cover_weighted(v, k, t, rng):
    """Weighted greedy construction: prioritize covering rare (hard-to-cover) triples."""
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    block_set_used = set()
    
    # Precompute all (t-1)-subsets for each point to speed up gain calculation
    # For each point x and each (t-1)-set S not containing x, precompute the t-set S∪{x}
    
    while uncovered:
        # Pick the uncovered t-subset with minimum "coverage potential" (hardest to cover)
        # Approximate by picking randomly but with bias toward less-covered points
        seed_list = list(uncovered)
        seed = rng.choice(seed_list)
        block_set = set(seed)
        
        # Greedily add points to maximize weighted coverage
        while len(block_set) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_set]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block_set, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        # Weight by how "rare" this triple is (fewer other blocks cover it)
                        gain += 1.0
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block_set.add(best_x)
            
        block = tuple(sorted(block_set))
        if block not in block_set_used:
            block_set_used.add(block)
            blocks.append(block)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
                
    return blocks


def _prune_redundant(blocks, v, k, t, rng):
    """Remove redundant blocks iteratively. Returns pruned list."""
    # Compute coverage counts
    counts = {}
    for b in blocks:
        for fs in itertools.combinations(b, t):
            f = frozenset(fs)
            counts[f] = counts.get(f, 0) + 1
    
    changed = True
    while changed:
        changed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for fs in itertools.combinations(b, t):
                    f = frozenset(fs)
                    counts[f] -= 1
                    if counts[f] == 0:
                        del counts[f]
                blocks.pop(i)
                changed = True
                break
    
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Build initial valid covering using weighted greedy
    blocks = _greedy_cover_weighted(v, k, t, rng)
    
    # Prune redundant blocks from greedy solution
    blocks = _prune_redundant(blocks, v, k, t, rng)
    
    def compute_counts(blk_list):
        counts = {}
        for b in blk_list:
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                counts[f] = counts.get(f, 0) + 1
        return counts

    counts = compute_counts(blocks)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    total_t = len(list(itertools.combinations(range(v), t)))
    
    def get_uncovered_count():
        return sum(1 for c in itertools.combinations(range(v), t) if counts.get(frozenset(c), 0) == 0)
    
    uncovered = get_uncovered_count()
    
    def apply_block_change(blk, sign):
        """
        sign = 1 to add, -1 to remove.
        Returns change in uncovered count (delta_uncovered).
        """
        delta_uncovered = 0
        for fs in itertools.combinations(blk, t):
            f = frozenset(fs)
            old_c = counts.get(f, 0)
            new_c = old_c + sign
            if new_c == 0:
                if f in counts:
                    del counts[f]
                if old_c == 1:
                    delta_uncovered += 1
                if old_c == 0:
                    delta_uncovered -= 1
            elif new_c > 0:
                if old_c == 0:
                    delta_uncovered -= 1
                counts[f] = new_c
        return delta_uncovered

    def try_remove_redundant():
        """Try to remove redundant blocks. Returns True if any removed."""
        nonlocal uncovered, best_blocks, best_count
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                delta_u = apply_block_change(b, -1)
                new_uncovered = uncovered + delta_u
                if new_uncovered == 0:
                    blocks.pop(i)
                    uncovered = new_uncovered
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed_any = True
                    break
                else:
                    apply_block_change(b, 1)
                    uncovered = uncovered  # unchanged
        return removed_any

    def try_multi_remove():
        """Try to remove multiple redundant blocks in one pass."""
        nonlocal uncovered, best_blocks, best_count
        # Find all currently redundant blocks
        redundant_indices = []
        for i in range(len(blocks)):
            b = blocks[i]
            redundant = True
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                redundant_indices.append(i)
        
        if not redundant_indices:
            return False
        
        # Try removing them in random order, keeping track of what's still redundant
        removed_any = False
        for i in sorted(redundant_indices, reverse=True):
            b = blocks[i]
            # Re-check redundancy after previous removals
            still_redundant = True
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                if counts.get(f, 0) <= 1:
                    still_redundant = False
                    break
            if still_redundant:
                delta_u = apply_block_change(b, -1)
                new_uncovered = uncovered + delta_u
                if new_uncovered == 0:
                    blocks.pop(i)
                    uncovered = new_uncovered
                    removed_any = True
                else:
                    apply_block_change(b, 1)
        
        if removed_any and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
        return removed_any

    def try_swap_block(i):
        """Try to replace block i with a better one. Returns True if improved."""
        nonlocal uncovered, best_blocks, best_count
        old_block = blocks[i]
        
        # Find critical t-subsets (covered only by this block)
        critical = []
        for fs in itertools.combinations(old_block, t):
            f = frozenset(fs)
            if counts.get(f, 0) == 1:
                critical.append(f)
        
        if not critical:
            # Can remove this block
            delta_u = apply_block_change(old_block, -1)
            new_uncovered = uncovered + delta_u
            if new_uncovered == 0:
                blocks.pop(i)
                uncovered = new_uncovered
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                return True
            else:
                apply_block_change(old_block, 1)
                uncovered = uncovered
                return False
        
        # Need to find a new block that covers all critical t-subsets
        points_needed = set()
        for fs in critical:
            points_needed.update(fs)
        
        if len(points_needed) > k:
            return False
        
        # Generate candidate new blocks
        new_block_set = set(points_needed)
        cands = [x for x in range(v) if x not in new_block_set]
        rng.shuffle(cands)
        
        # Try to fill the rest with points that maximize coverage
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
                delta_u_remove = apply_block_change(old_block, -1)
                delta_u_add = apply_block_change(new_block, 1)
                
                new_uncovered = uncovered + delta_u_remove + delta_u_add
                
                if new_uncovered == 0:
                    blocks[i] = new_block
                    uncovered = new_uncovered
                    return True
                else:
                    apply_block_change(new_block, -1)
                    apply_block_change(old_block, 1)
                    uncovered = uncovered
        return False

    def perturb():
        """Perturbation: remove a block and re-add to escape local minima."""
        nonlocal blocks, counts, uncovered, best_blocks, best_count
        
        old_blocks = list(blocks)
        old_counts = dict(counts)
        old_uncovered = uncovered
        
        i = rng.randrange(len(blocks))
        b = blocks[i]
        
        delta_u = apply_block_change(b, -1)
        uncovered += delta_u
        blocks.pop(i)
        
        # Re-add blocks to cover uncovered t-subsets
        unc_list = []
        for c in itertools.combinations(range(v), t):
            f = frozenset(c)
            if counts.get(f, 0) == 0:
                unc_list.append(f)
        
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
            
            if new_b_tuple not in blocks:
                blocks.append(new_b_tuple)
                delta_u_add = apply_block_change(new_b_tuple, 1)
                uncovered += delta_u_add
                
                unc_list = []
                for c in itertools.combinations(range(v), t):
                    f = frozenset(c)
                    if counts.get(f, 0) == 0:
                        unc_list.append(f)
        
        if uncovered == 0:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
            return True
        else:
            blocks = old_blocks
            counts = old_counts
            uncovered = old_uncovered
            return False

    # Simulated annealing parameters
    initial_temp = 1.0
    cooling_rate = 0.9995
    min_temp = 0.001
    
    temp = initial_temp
    no_improve = 0
    max_no_improve = 300
    
    while time.time() - start_time < time_limit and temp > min_temp:
        
        # 1. Try to remove redundant blocks (multi-remove first)
        if try_multi_remove():
            no_improve = 0
            continue
        if try_remove_redundant():
            no_improve = 0
            continue
            
        # 2. Try swaps
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            if try_swap_block(i):
                no_improve = 0
                continue
        
        # 3. Perturbation with simulated annealing acceptance
        if no_improve > max_no_improve:
            success = perturb()
            if success:
                no_improve = 0
            else:
                # Accept with probability based on temperature
                if rng.random() < math.exp(-1.0 / max(temp, 0.001)):
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            no_improve += 1
        
        temp *= cooling_rate
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
