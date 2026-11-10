"""Covering design C(v, k, t) – randomized greedy with deadline guard."""
import itertools
import time
import random

_DEADLINE_S = 112.0  # Safety margin from 120s evaluator limit
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _greedy_block(uncovered, v, k, t, seed):
    """Build a k-subset containing seed, maximizing coverage of uncovered t-subsets."""
    block = list(seed)
    block_set = set(block)
    while len(block) < k and _time_left() > 0.3:
        best_pt = None
        best_gain = -1
        for x in range(v):
            if x in block_set:
                continue
            gain = 0
            for combo in itertools.combinations(block, t - 1):
                sub = tuple(sorted(combo + (x,)))
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_pt = x
        if best_pt is None:
            candidates = [x for x in range(v) if x not in block_set]
            if not candidates:
                break
            best_pt = random.choice(candidates)
        block.append(best_pt)
        block_set.add(best_pt)
    # Fill if needed
    if len(block) < k:
        for x in range(v):
            if len(block) >= k:
                break
            if x not in block_set:
                block.append(x)
                block_set.add(x)
    return sorted(block)


def construct(v, k, t):
    random.seed(12345)
    all_tsubs = list(itertools.combinations(range(v), t))
    best_blocks = None
    best_len = float('inf')
    
    # Multiple randomized restarts
    max_restarts = 30
    for r in range(max_restarts):
        if _time_left() < 1.5:
            break
        uncovered = set(all_tsubs)
        blocks = []
        block_set = set()
        
        while uncovered and _time_left() > 0.5:
            # Pick target: prefer uncovered subsets with high "constraint" 
            # (few points that appear in already-chosen blocks? Just use random for diversity)
            if len(uncovered) > 200:
                sample = random.sample(list(uncovered), min(100, len(uncovered)))
                target = random.choice(sample)
            else:
                target = random.choice(list(uncovered))
            
            block = _greedy_block(uncovered, v, k, t, seed=target)
            bkey = tuple(block)
            if bkey not in block_set:
                blocks.append(block)
                block_set.add(bkey)
                for sub in itertools.combinations(block, t):
                    uncovered.discard(sub)
        
        if not uncovered:
            if len(blocks) < best_len:
                best_blocks = blocks
                best_len = len(blocks)
                if best_len <= 151:  # Known best
                    break
    
    # Fallback if no valid covering found in time
    if best_blocks is None:
        uncovered = set(all_tsubs)
        blocks = []
        while uncovered:
            target = min(uncovered)
            block = _greedy_block(uncovered, v, k, t, seed=target)
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        best_blocks = blocks
    
    return best_blocks
