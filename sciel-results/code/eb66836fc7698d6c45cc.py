"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


# Pre-compute all 3-subsets of 7-element blocks for v=24, k=7, t=3
# We'll use a dictionary mapping frozenset(block) -> list of frozenset(t-subsets)
_block_triple_cache = {}

def _get_block_triples(block_tuple, t=3):
    """Get all t-subsets of a k-block, with caching."""
    if block_tuple not in _block_triple_cache:
        _block_triple_cache[block_tuple] = [frozenset(c) for c in itertools.combinations(block_tuple, t)]
    return _block_triple_cache[block_tuple]


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    # Track uncovered triples
    uncovered = set(frozenset(c) for c in itertools.combinations(range(v), t))
    blocks = []
    block_set_seen = set()
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered))
        block_set = set(seed)
        
        # Greedily add points to maximize coverage of *uncovered* subsets
        while len(block_set) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_set]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                # Check all (t-1) subsets of current block
                for rest in itertools.combinations(block_set, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                # Tie-breaking: random
                if gain > best_gain or (gain == best_gain and best_x is not None and rng.random() < 0.5):
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block_set.add(best_x)
            
        block = tuple(sorted(block_set))
        if block not in block_set_seen:
            block_set_seen.add(block)
            blocks.append(block)
            for c in _get_block_triples(block, t):
                uncovered.discard(c)
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    
    def compute_counts(blk_list):
        counts = defaultdict(int)
        for b in blk_list:
            for fs in _get_block_triples(b, t):
                counts[fs] += 1
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
        for f in _get_block_triples(blk, t):
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

    def try_rebuild_block(i):
        """
        Try to replace block i with a new block that covers all critical subsets
        of the old block and potentially more uncovered subsets.
        Returns True if successful (valid covering maintained).
        """
        old_block = blocks[i]
        
        # Identify critical subsets (those covered only by this block)
        critical = []
        for f in _get_block_triples(old_block, t):
            if counts.get(f, 0) == 1:
                critical.append(f)
        
        # If no critical subsets, the block is redundant (handled by removal step usually)
        if not critical:
            apply_block_change(old_block, 1)
            return False

        # Temporarily remove old block
        apply_block_change(old_block, -1)
        
        # must_cover = critical sets that are now uncovered
        must_cover = set(critical)
        
        # Greedily construct a new block to cover must_cover
        # Start with a random subset from must_cover
        seed = rng.choice(list(must_cover))
        new_block_set = set(seed)
        
        # We need to cover all sets in must_cover.
        # Greedy: at each step, add the point that covers the most uncovered sets in must_cover
        
        while len(new_block_set) < k:
            best_x = None
            best_score = -1
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            for x in cands:
                # Count how many sets in must_cover are covered by adding x
                gain_must = 0
                for rest in itertools.combinations(new_block_set, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in must_cover:
                        gain_must += 1
                
                # Tie-breaking: random
                if gain_must > best_score or (gain_must == best_score and best_x is not None and rng.random() < 0.5):
                    best_score = gain_must
                    best_x = x
            
            if best_x is None:
                break
            new_block_set.add(best_x)
        
        new_block = tuple(sorted(new_block_set))
        
        # Check if new_block is not the same as old_block
        if new_block == old_block:
            apply_block_change(old_block, 1)
            return False
            
        # Verify coverage of must_cover
        covered_must = set()
        for f in _get_block_triples(new_block, t):
            if f in must_cover:
                covered_must.add(f)
        
        if covered_must != must_cover:
            # Failed to cover all critical subsets
            apply_block_change(old_block, 1)
            return False
        
        # If we reach here, new_block covers all critical subsets.
        # We can replace old_block with new_block.
        # Apply add for new_block
        apply_block_change(new_block, 1)
        
        # Update blocks list
        blocks[i] = new_block
        
        return True

    no_improve_iter = 0
    max_no_improve = 200
    restart_count = 0
    max_restarts = 10
    
    while time.time() - start_time < time_limit:
        
        # 1. Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for f in _get_block_triples(b, t):
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                apply_block_change(b, -1)
                blocks.pop(i)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                removed_any = True
                no_improve_iter = 0
                break
        
        if removed_any:
            continue
            
        # 2. Try Rebuild (Replacement)
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            # Only try rebuild if the block has critical subsets
            has_critical = False
            for f in _get_block_triples(blocks[i], t):
                if counts.get(f, 0) == 1:
                    has_critical = True
                    break
            
            if has_critical:
                success = try_rebuild_block(i)
                if success:
                    no_improve_iter = 0
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                else:
                    no_improve_iter += 1
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1

        # 3. Perturbation if stuck
        if no_improve_iter > max_no_improve:
            restart_count += 1
            if restart_count > max_restarts:
                break
            
            # Restart from best solution with different random seed
            blocks = list(best_blocks)
            counts = compute_counts(blocks)
            no_improve_iter = 0
            # Change random seed slightly
            rng.seed(rng.randint(0, 2**31))

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
