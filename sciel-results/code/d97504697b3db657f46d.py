"""Randomized local search for covering design C(24,7,3) with index-based state management."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
from collections import defaultdict


def _precompute(v, k, t):
    """Precompute subset indices and per-point subset lists."""
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # For each point, list of subset indices containing it
    point_to_subsets = [[] for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        for p in s:
            point_to_subsets[p].append(idx)
    
    return all_subsets, subset_to_idx, point_to_subsets, n_subsets


def _block_subsets_indices(block, subset_to_idx, t):
    """Return list of t-subset indices covered by a block."""
    res = []
    for c in itertools.combinations(block, t):
        res.append(subset_to_idx[frozenset(c)])
    return res


def _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, point_to_subsets, n_subsets):
    """Dynamic scarcity-weighted greedy construction for a tight initial covering."""
    uncovered = set(range(n_subsets))
    # Scarcity: for each point, how many uncovered subsets contain it (lower = more scarce)
    # We'll maintain point_scarcity dynamically
    point_scarcity = [0] * v
    for idx in uncovered:
        for p in all_subsets[idx]:
            point_scarcity[p] += 1
    
    blocks = []
    block_set_seen = set()
    
    while uncovered:
        # Pick the most scarce uncovered subset (min sum of point scarcities, tie-break random)
        min_scarcity = float('inf')
        candidates = []
        for idx in uncovered:
            s = sum(point_scarcity[p] for p in all_subsets[idx])
            if s < min_scarcity:
                min_scarcity = s
                candidates = [idx]
            elif s == min_scarcity:
                candidates.append(idx)
        seed_idx = rng.choice(candidates)
        seed = all_subsets[seed_idx]
        
        block_set = set(seed)
        
        # Greedily add points to maximize coverage of uncovered subsets, weighted by scarcity
        while len(block_set) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_set]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block_set, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in subset_to_idx:
                        idx = subset_to_idx[fs]
                        if idx in uncovered:
                            # Weight by scarcity of the triple (rarer triples are harder)
                            triple_scarcity = sum(point_scarcity[p] for p in all_subsets[idx])
                            gain += (v * 10 - triple_scarcity)
                if gain > best_gain or (gain == best_gain and best_x is not None and rng.random() < 0.3):
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block_set.add(best_x)
            
        block = tuple(sorted(block_set))
        if block not in block_set_seen:
            block_set_seen.add(block)
            blocks.append(block)
            for idx in _block_subsets_indices(block, subset_to_idx, t):
                if idx in uncovered:
                    uncovered.discard(idx)
                    for p in all_subsets[idx]:
                        point_scarcity[p] -= 1
                        
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute
    all_subsets, subset_to_idx, point_to_subsets, n_subsets = _precompute(v, k, t)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, point_to_subsets, n_subsets)
    
    # State management with integer indices
    block_covered = [_block_subsets_indices(b, subset_to_idx, t) for b in blocks]
    counts = [0] * n_subsets
    for covered in block_covered:
        for idx in covered:
            counts[idx] += 1
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    def try_remove_redundant():
        """Try to remove a redundant block. Returns True if removed."""
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            redundant = True
            for idx in block_covered[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                return True
        return False
    
    def try_swap_block(i):
        """
        Try to replace block i with a new block that covers all critical subsets.
        Returns True if successful.
        """
        old_block = blocks[i]
        old_covered = block_covered[i]
        
        # Identify critical subsets (count == 1)
        critical = [idx for idx in old_covered if counts[idx] == 1]
        
        if not critical:
            # No critical subsets: block is redundant, but we handle that separately
            return False
        
        # Points that MUST be in the new block
        points_needed = set()
        for idx in critical:
            points_needed.update(all_subsets[idx])
        
        if len(points_needed) > k:
            return False
        
        # Temporarily remove old block
        for idx in old_covered:
            counts[idx] -= 1
        
        # Build new block starting with points_needed
        new_block_set = set(points_needed)
        
        # Greedily fill remaining slots to maximize coverage of low-count subsets
        remaining_slots = k - len(new_block_set)
        cands = [x for x in range(v) if x not in new_block_set]
        rng.shuffle(cands)
        
        for _ in range(remaining_slots):
            if not cands:
                break
            best_x = None
            best_score = -1
            
            for x in cands:
                score = 0
                # Count how many subsets this would cover with low counts
                for rest in itertools.combinations(new_block_set, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in subset_to_idx:
                        idx = subset_to_idx[fs]
                        if counts[idx] == 0:
                            score += 100  # Critical: covers an uncovered subset
                        elif counts[idx] == 1:
                            score += 10   # Helps with another critical
                        else:
                            score += 1
                if score > best_score or (score == best_score and best_x is not None and rng.random() < 0.5):
                    best_score = score
                    best_x = x
            
            if best_x is None:
                # Fill with random
                best_x = cands[0]
            
            new_block_set.add(best_x)
            cands.remove(best_x)
        
        new_block = tuple(sorted(new_block_set))
        
        if new_block == old_block:
            # Restore
            for idx in old_covered:
                counts[idx] += 1
            return False
        
        # Check validity: all critical subsets must be covered
        new_covered = _block_subsets_indices(new_block, subset_to_idx, t)
        new_covered_set = set(new_covered)
        
        valid = True
        for idx in critical:
            if idx not in new_covered_set:
                valid = False
                break
        
        if not valid:
            # Restore
            for idx in old_covered:
                counts[idx] += 1
            return False
        
        # Check no other subset loses its last cover
        for idx in old_covered:
            if idx not in new_covered_set and counts[idx] <= 0:
                valid = False
                break
        
        if not valid:
            for idx in old_covered:
                counts[idx] += 1
            return False
        
        # Apply swap
        for idx in new_covered:
            counts[idx] += 1
        
        blocks[i] = new_block
        block_covered[i] = new_covered
        return True
    
    def try_perturb():
        """
        Perturbation: temporarily add a random block to escape local minima,
        then try to remove blocks.
        """
        # Add a random block
        new_block = tuple(sorted(rng.sample(range(v), k)))
        new_covered = _block_subsets_indices(new_block, subset_to_idx, t)
        
        blocks.append(new_block)
        block_covered.append(new_covered)
        for idx in new_covered:
            counts[idx] += 1
        
        # Try to remove redundant blocks (including the one we just added)
        removed = 0
        for _ in range(5):
            if not try_remove_redundant():
                break
            removed += 1
        
        return removed
    
    no_improve = 0
    max_no_improve = 150
    restart_count = 0
    max_restarts = 15
    
    while time.time() - start_time < time_limit:
        # 1. Try to remove redundant blocks
        if try_remove_redundant():
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
            no_improve = 0
            continue
        
        # 2. Try to swap a block
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            if try_swap_block(i):
                no_improve = 0
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # 3. Perturbation if stuck
        if no_improve > max_no_improve:
            restart_count += 1
            if restart_count > max_restarts:
                break
            
            # Restart from best solution
            blocks = [tuple(b) for b in best_blocks]
            block_covered = [_block_subsets_indices(b, subset_to_idx, t) for b in blocks]
            counts = [0] * n_subsets
            for covered in block_covered:
                for idx in covered:
                    counts[idx] += 1
            no_improve = 0
            rng.seed(rng.randint(0, 2**31))
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
