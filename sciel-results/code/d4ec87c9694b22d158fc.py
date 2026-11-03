"""Stronger covering design C(19,7,4): fast greedy + local search with deadline guard."""
import time
import itertools
import numpy as np
from collections import defaultdict

_DEADLINE = 140.0  # seconds wall-clock budget (kill at 150)
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _all_tsubs(v, t):
    return list(itertools.combinations(range(v), t))

def _subsets_of_block(block, t):
    return list(itertools.combinations(block, t))

def _bitset_from_blocks(blocks, v, t):
    """Return a set of frozensets of t-subsets covered by blocks."""
    covered = set()
    for b in blocks:
        for s in itertools.combinations(b, t):
            covered.add(frozenset(s))
    return covered

def _uncovered_set(all_tsubs, covered):
    return set(frozenset(s) for s in all_tsubs if frozenset(s) not in covered)

def _greedy_construct(v, k, t, deadline_check):
    """Fast greedy: pick block maximizing newly-covered t-subsets."""
    all_tsubs = _all_tsubs(v, t)
    # Represent each t-subset by a frozenset for O(1) lookup
    uncovered = set(frozenset(s) for s in all_tsubs)
    blocks = []
    points = list(range(v))
    
    while uncovered and deadline_check():
        # Start with the first uncovered t-subset (lexicographically smallest)
        first = min(uncovered)
        block = list(first)
        
        while len(block) < k and deadline_check():
            block_set = set(block)
            # For each candidate point, compute how many uncovered t-subsets it would cover
            # A t-subset {a1,...,at} is covered by block+{x} iff it contains x and the rest is subset of block
            # i.e., t-subset = {x} union (t-1)-subset of block
            best_x = None
            best_gain = -1
            # Precompute (t-1)-subsets of current block
            block_t1 = list(itertools.combinations(block, t - 1))
            for x in points:
                if x in block_set:
                    continue
                gain = 0
                for sub in block_t1:
                    s = frozenset(sub) | {x}
                    if s in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(tuple(block))
        # Remove all t-subsets of this block from uncovered
        for s in itertools.combinations(block, t):
            uncovered.discard(frozenset(s))
    
    return [list(b) for b in blocks]

def _local_search(blocks, v, k, t, all_tsubs, deadline_check, max_iters=5000):
    """Try to reduce block count via replacement and swap moves."""
    blocks = [list(b) for b in blocks]
    # Precompute coverage: which blocks cover each t-subset
    # For speed, track count of blocks covering each t-subset
    sub_to_blocks = defaultdict(set)
    for i, b in enumerate(blocks):
        for s in itertools.combinations(b, t):
            sub_to_blocks[frozenset(s)].add(i)
    
    all_tsubs_f = [frozenset(s) for s in all_tsubs]
    
    def is_covered():
        for s in all_tsubs_f:
            if len(sub_to_blocks[s]) == 0:
                return False
        return True
    
    if not is_covered():
        return blocks
    
    improved = True
    iter_count = 0
    while improved and deadline_check() and iter_count < max_iters:
        improved = False
        iter_count += 1
        n = len(blocks)
        # Try removing a redundant block
        for i in range(n):
            if not deadline_check():
                break
            # Check if block i is redundant
            redundant = True
            for s in itertools.combinations(blocks[i], t):
                fs = frozenset(s)
                if len(sub_to_blocks[fs]) <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for s in itertools.combinations(blocks[i], t):
                    fs = frozenset(s)
                    sub_to_blocks[fs].discard(i)
                blocks.pop(i)
                # Reindex
                for fs in sub_to_blocks:
                    new_set = set()
                    for j in sub_to_blocks[fs]:
                        new_set.add(j - 1 if j > i else j)
                    sub_to_blocks[fs] = new_set
                improved = True
                break
        
        if improved:
            continue
        
        # Try replacing a block with a better one
        # For each block, try swapping one point
        for i in range(len(blocks)):
            if not deadline_check():
                break
            block = blocks[i]
            for pos in range(len(block)):
                for x in range(v):
                    if x in block:
                        continue
                    new_block = block[:]
                    new_block[pos] = x
                    new_block.sort()
                    # Check: does removing old block and adding new block keep all covered?
                    # First, check which t-subsets would lose their last cover
                    old_subs = set(frozenset(s) for s in itertools.combinations(block, t))
                    new_subs = set(frozenset(s) for s in itertools.combinations(new_block, t))
                    lost = old_subs - new_subs
                    ok = True
                    for fs in lost:
                        if len(sub_to_blocks[fs]) <= 1:
                            ok = False
                            break
                    if ok:
                        # Apply the change
                        for fs in old_subs:
                            sub_to_blocks[fs].discard(i)
                        for fs in new_subs:
                            sub_to_blocks[fs].add(i)
                        blocks[i] = new_block
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return blocks

def construct(v, k, t):
    global _START
    _START = time.time()
    
    all_tsubs = _all_tsubs(v, t)
    
    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t, lambda: _time_left() > 5.0)
    
    # Phase 2: Local search refinement
    if _time_left() > 10.0:
        blocks = _local_search(blocks, v, k, t, all_tsubs, lambda: _time_left() > 5.0, max_iters=10000)
    
    # Ensure distinct blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        tkey = tuple(sorted(b))
        if tkey not in seen:
            seen.add(tkey)
            unique_blocks.append(list(tkey))
    
    return unique_blocks
