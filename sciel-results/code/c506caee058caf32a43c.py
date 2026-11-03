"""Simulated Annealing for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _precompute_triple_indices(v, t):
    """Map each t-subset to a unique index."""
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    return triples, triple_to_idx


def _greedy_cover(v, k, t, rng, triples, triple_to_idx):
    """Fast greedy construction using bitmask for uncovered set."""
    all_points = list(range(v))
    num_triples = len(triples)
    # Initialize mask with all bits set
    uncovered_mask = (1 << num_triples) - 1
    blocks = []
    
    while uncovered_mask:
        # Pick a random uncovered triple
        seed_idx = rng.randrange(num_triples)
        while not (uncovered_mask & (1 << seed_idx)):
            seed_idx = rng.randrange(num_triples)
            
        seed_triple = triples[seed_idx]
        block = list(seed_triple)
        rng.shuffle(block)
        
        # Greedily add points
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in all_points if x not in block]
            rng.shuffle(cands)
            
            # Precompute pairs of current block for t=3
            if t == 3:
                pairs = list(itertools.combinations(block, 2))
                for x in cands:
                    gain = 0
                    for p in pairs:
                        s = tuple(sorted(p + (x,)))
                        idx = triple_to_idx.get(s)
                        if idx is not None and (uncovered_mask & (1 << idx)):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            else:
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        idx = triple_to_idx.get(s)
                        if idx is not None and (uncovered_mask & (1 << idx)):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in blocks:
            blocks.append(b_tuple)
            # Update uncovered_mask
            for c in itertools.combinations(block, t):
                idx = triple_to_idx.get(tuple(sorted(c)))
                if idx is not None:
                    uncovered_mask &= ~(1 << idx)
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 118.0  # leave 2s buffer

    triples, triple_to_idx = _precompute_triple_indices(v, t)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, triples, triple_to_idx)
    
    def compute_counts(blk_list):
        counts = {}
        for b in blk_list:
            for fs in itertools.combinations(b, t):
                f = tuple(sorted(fs))
                idx = triple_to_idx[f]
                counts[idx] = counts.get(idx, 0) + 1
        return counts

    counts = compute_counts(blocks)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    def apply_block_change(blk, sign):
        """
        sign = 1 to add, -1 to remove.
        Returns change in uncovered count (delta_uncovered).
        """
        delta_uncovered = 0
        for fs in itertools.combinations(blk, t):
            f = tuple(sorted(fs))
            idx = triple_to_idx[f]
            old_c = counts.get(idx, 0)
            new_c = old_c + sign
            if new_c == 0:
                if idx in counts:
                    del counts[idx]
                if old_c == 1:
                    delta_uncovered += 1
                elif old_c == 0:
                    delta_uncovered -= 1
            elif new_c > 0:
                if old_c == 0:
                    delta_uncovered -= 1
                counts[idx] = new_c
        return delta_uncovered

    def try_remove_block(index):
        """
        Try to remove block at index. Returns True if successful (improvement).
        """
        b = blocks[index]
        redundant = True
        for fs in itertools.combinations(b, t):
            f = tuple(sorted(fs))
            idx = triple_to_idx[f]
            if counts.get(idx, 0) <= 1:
                redundant = False
                break
        
        if redundant:
            delta_u = apply_block_change(b, -1)
            # delta_u should be 0 if redundant
            if delta_u == 0:
                blocks.pop(index)
                return True
            else:
                apply_block_change(b, 1)
        return False

    def try_swap_block(index, new_block):
        """
        Try to replace block at index with new_block.
        Returns True if successful (valid swap).
        """
        old_block = blocks[index]
        if old_block == new_block:
            return False
        
        # Check if new_block is already in blocks
        for i, b in enumerate(blocks):
            if i != index and b == new_block:
                return False

        delta_remove = apply_block_change(old_block, -1)
        
        delta_add = apply_block_change(new_block, 1)
        
        total_delta = delta_remove + delta_add
        
        if total_delta == 0:
            # Valid swap
            blocks[index] = new_block
            return True
        else:
            # Revert
            apply_block_change(new_block, -1)
            apply_block_change(old_block, 1)
            return False

    no_improve_iter = 0
    max_no_improve = 100
    temp = 1.0
    cooling = 0.999

    while time.time() - start_time < time_limit:
        
        # 1. Try to remove redundant blocks
        # Pick a few random indices
        for _ in range(5):
            if not blocks: break
            i = rng.randrange(len(blocks))
            if try_remove_block(i):
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                no_improve_iter = 0
                break
        
        if no_improve_iter == 0:
            continue

        # 2. Try swaps
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Generate a new block similar to old_block
            keep_size = rng.randint(2, k-1)
            keep_points = rng.sample(old_block, keep_size)
            
            new_block_set = set(keep_points)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            # Fill rest randomly
            while len(new_block_set) < k:
                new_block_set.add(cands.pop())
            
            new_block = tuple(sorted(new_block_set))
            
            if try_swap_block(i, new_block):
                no_improve_iter = 0
                continue

        # 3. Perturbation if stuck
        if no_improve_iter > max_no_improve:
            # Destroy and Repair
            old_blocks = list(blocks)
            old_counts = dict(counts)
            
            # Remove a random block
            i = rng.randrange(len(blocks))
            b = blocks[i]
            apply_block_change(b, -1)
            blocks.pop(i)
            
            # Find uncovered sets
            unc_list = []
            for idx, c in counts.items():
                if c == 0:
                    unc_list.append(idx)
            
            if len(unc_list) > 0:
                # Greedily cover uncovered sets
                seed_idx = rng.choice(unc_list)
                seed = triples[seed_idx]
                new_block = list(seed)
                rng.shuffle(new_block)
                
                current_unc = set(unc_list)
                
                while len(new_block) < k and current_unc:
                    best_x = None
                    best_gain = -1
                    cands = [x for x in range(v) if x not in new_block]
                    rng.shuffle(cands)
                    
                    for x in cands:
                        gain = 0
                        for rest in itertools.combinations(new_block, t-1):
                            fs = tuple(sorted(rest + (x,)))
                            idx = triple_to_idx.get(fs)
                            if idx is not None and idx in current_unc:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    
                    if best_x is None or best_gain == 0:
                        break
                    new_block.append(best_x)
                    # Update current_unc
                    for rest in itertools.combinations(new_block, t):
                        idx = triple_to_idx.get(tuple(sorted(rest)))
                        if idx is not None:
                            current_unc.discard(idx)
                
                new_block.sort()
                new_block = tuple(new_block)
                
                # Check if it covers all original unc_list
                apply_block_change(new_block, 1)
                
                # Check if any uncovered remain
                still_unc = 0
                for idx, c in counts.items():
                    if c == 0:
                        still_unc += 1
                        if still_unc > 0: break # just need to know if >0
                
                if still_unc == 0:
                    blocks.append(new_block)
                    no_improve_iter = 0
                else:
                    # Failed, revert
                    apply_block_change(new_block, -1)
                    blocks = old_blocks
                    counts = old_counts
                    no_improve_iter += 1
            else:
                # No uncovered sets? That means the removed block was redundant.
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                no_improve_iter = 0

        else:
            no_improve_iter += 1

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
