"""Portfolio search for covering design C(24,7,3)."""
import itertools
import random
import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
import numpy as np

def _greedy_covering(v, k, t, rng, time_limit, mode='basic', diversity_bonus=0.0):
    """
    Greedy covering with various scoring modes.
    mode='basic': pure gain
    mode='pair': gain + diversity bonus based on pair rarity
    mode='adaptive': gain with adaptive weight on diversity
    """
    deadline = time.time() + time_limit
    
    # All t-sets
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    # Track pair block counts for diversity bonus
    pair_block_count = defaultdict(int)
    
    def compute_gain(x, block):
        """Number of uncovered t-sets that contain x and (t-1) elements of block."""
        cnt = 0
        for S in itertools.combinations(block, t - 1):
            if x not in S:
                tset = tuple(sorted(S + (x,)))
                if tset in uncovered:
                    cnt += 1
        return cnt
    
    def compute_diversity_bonus(x, block):
        """Bonus based on how rare the pairs (x, y) are for y in block."""
        bonus = 0.0
        for y in block:
            p = (min(x, y), max(x, y))
            cnt = pair_block_count[p]
            bonus += 1.0 / (1.0 + cnt)
        return bonus
    
    while uncovered and time.time() < deadline:
        # Start with the lexicographically smallest uncovered t-set
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            
            best_x = None
            best_score = -1
            
            for x in candidates:
                gain = compute_gain(x, block)
                
                if mode == 'basic':
                    score = float(gain)
                elif mode == 'pair':
                    bonus = compute_diversity_bonus(x, block)
                    score = gain + diversity_bonus * bonus
                elif mode == 'adaptive':
                    bonus = compute_diversity_bonus(x, block)
                    # Weight diversity more when gain is low relative to remaining work
                    adaptive_w = max(0.1, 1.0 - gain / max(1, len(uncovered)))
                    score = gain + adaptive_w * bonus
                else:
                    score = float(gain)
                
                # Small random tie-breaker for diversity
                score += rng.random() * 0.01
                
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update pair counts
        for p in itertools.combinations(block, 2):
            pair_block_count[p] += 1
        
        # Remove covered t-sets
        covered = set(itertools.combinations(block, t))
        uncovered -= covered
    
    return [list(b) for b in blocks]


def _build_tset_index(blocks, v, t):
    """Build a dictionary mapping each t-set to the list of block indices covering it."""
    tset_to_blocks = defaultdict(list)
    for i, b in enumerate(blocks):
        bset = set(b)
        for ts in itertools.combinations(b, t):
            tset_to_blocks[ts].append(i)
    return tset_to_blocks


def _local_search(blocks, v, k, t, rng, time_limit):
    """
    Efficient local search using tset-to-blocks index.
    - Remove redundant blocks
    - Try to swap blocks to cover more t-sets
    """
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    
    if not blocks:
        return blocks
    
    # Build initial index
    tset_to_blocks = _build_tset_index(blocks, v, t)
    
    # Precompute all t-sets for validation
    all_tsets = list(itertools.combinations(range(v), t))
    
    def rebuild_index():
        nonlocal tset_to_blocks
        tset_to_blocks = _build_tset_index(blocks, v, t)
    
    # Phase 1: Remove redundant blocks (iterate until no more)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            # Check if any t-set in block i is covered ONLY by block i
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                covering = tset_to_blocks.get(ts, [])
                if covering == [i]:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for ts in itertools.combinations(blocks[i], t):
                    if ts in tset_to_blocks and i in tset_to_blocks[ts]:
                        tset_to_blocks[ts].remove(i)
                blocks.pop(i)
                # Reindex remaining blocks
                rebuild_index()
                changed = True
                break
    
    # Phase 2: Try swaps to improve coverage
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        if n == 0:
            break
        
        # Try each block position
        for idx in range(n):
            if time.time() >= deadline:
                break
            
            old_block = blocks[idx]
            old_set = set(old_block)
            
            # Find t-sets only covered by this block (critical t-sets)
            critical_tsets = []
            for ts in itertools.combinations(old_block, t):
                covering = tset_to_blocks.get(ts, [])
                if covering == [idx]:
                    critical_tsets.append(ts)
            
            if not critical_tsets:
                # Block is redundant, remove it
                for ts in itertools.combinations(old_block, t):
                    if ts in tset_to_blocks and idx in tset_to_blocks[ts]:
                        tset_to_blocks[ts].remove(idx)
                blocks.pop(idx)
                rebuild_index()
                improved = True
                break
            
            # Try to find a replacement that covers all critical t-sets
            best_new_block = None
            best_extra = -1
            
            # Generate candidate replacements
            # Strategy: start from old_block, try random modifications
            num_candidates = 30
            for _ in range(num_candidates):
                if time.time() >= deadline:
                    break
                
                # Create a candidate by sampling
                cand = sorted(rng.sample(range(v), k))
                cand_set = set(cand)
                
                # Check if candidate covers all critical t-sets
                covers_all = True
                for ts in critical_tsets:
                    if not set(ts).issubset(cand_set):
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # Count extra t-sets covered (t-sets in cand not covered by other blocks)
                extra = 0
                for ts in itertools.combinations(cand, t):
                    covering = tset_to_blocks.get(ts, [])
                    # Remove idx from consideration since we're replacing it
                    other_covering = [c for c in covering if c != idx]
                    if not other_covering:
                        # This t-set is only covered by the old block (or not at all)
                        # If it's not in critical_tsets, it was covered by old block and others
                        # Actually, if covering == [idx], it's in critical_tsets
                        # If covering is empty, it wasn't covered at all (shouldn't happen in valid covering)
                        # If covering has others, it's already covered
                        if not covering or covering == [idx]:
                            extra += 1
                
                if extra > best_extra:
                    best_extra = extra
                    best_new_block = cand
            
            if best_new_block is not None and best_extra > 0:
                # Replace block
                # Remove old block's coverage
                for ts in itertools.combinations(old_block, t):
                    if ts in tset_to_blocks and idx in tset_to_blocks[ts]:
                        tset_to_blocks[ts].remove(idx)
                # Add new block's coverage
                blocks[idx] = best_new_block
                for ts in itertools.combinations(best_new_block, t):
                    if ts not in tset_to_blocks:
                        tset_to_blocks[ts] = []
                    if idx not in tset_to_blocks[ts]:
                        tset_to_blocks[ts].append(idx)
                improved = True
                break
    
    return blocks


def _verify_covering(blocks, v, t):
    """Verify that every t-subset is covered by at least one block."""
    tset_to_blocks = _build_tset_index(blocks, v, t)
    for ts in itertools.combinations(range(v), t):
        if ts not in tset_to_blocks or not tset_to_blocks[ts]:
            return False
    return True


def construct(v, k, t):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, use 140s budget
    total_budget = 140.0
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy runs with different modes and seeds
    # Allocate ~50% for greedy, ~45% for local search, 5% margin
    greedy_budget = total_budget * 0.50
    ls_budget = total_budget * 0.45
    margin = total_budget * 0.05
    
    # Define greedy configurations: (mode, diversity_bonus, seed)
    configs = [
        ('basic', 0.0, 42),
        ('pair', 0.5, 137),
        ('adaptive', 0.0, 274),
        ('basic', 0.0, 411),
        ('pair', 0.3, 548),
        ('adaptive', 0.0, 685),
    ]
    
    per_greedy_time = greedy_budget / len(configs)
    
    for mode, div_bonus, seed in configs:
        elapsed = time.time() - start_time
        if elapsed > total_budget - ls_budget - margin:
            break
        
        rng = random.Random(seed)
        blocks = _greedy_covering(v, k, t, rng, per_greedy_time, mode=mode, diversity_bonus=div_bonus)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    # If we have time, try a few more greedy runs with shorter time
    elapsed = time.time() - start_time
    remaining_for_more = total_budget - ls_budget - margin - elapsed
    if remaining_for_more > 5.0:
        # Try a couple more quick greedy runs
        for seed in [722, 859]:
            if time.time() - start_time > total_budget - ls_budget - margin:
                break
            rng = random.Random(seed)
            blocks = _greedy_covering(v, k, t, rng, 5.0, mode='adaptive', diversity_bonus=0.0)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
    
    if best_blocks is None:
        # Fallback
        blocks = _greedy_covering(v, k, t, random.Random(1), 5.0)
        best_blocks = blocks
        best_len = len(blocks)
    
    # Phase 2: Local search refinement
    remaining = total_budget - (time.time() - start_time) - margin
    if remaining > 10.0 and best_blocks is not None:
        rng = random.Random(12345)
        refined = _local_search(best_blocks, v, k, t, rng, remaining)
        
        # Verify the refined covering is valid
        if _verify_covering(refined, v, t):
            if len(refined) < best_len:
                best_blocks = refined
                best_len = len(refined)
    
    # Final safety check
    if time.time() - start_time > total_budget - 5.0:
        return best_blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
