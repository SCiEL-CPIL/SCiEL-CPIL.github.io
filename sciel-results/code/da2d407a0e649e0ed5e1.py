"""Improved greedy covering with randomized restarts, pivot-based block selection,
and a deletion phase to remove redundant blocks. Uses explicit deadline guard."""
import itertools
import random
import time
from collections import Counter

# Precomputation cache keyed by (v, k, t)
_CACHE = {}

def _precompute(v, k, t):
    """Precompute t-subsets of each block and reverse index."""
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_tsubs = []
    for b in all_blocks:
        block_tsubs.append(tuple(sorted(c) for c in itertools.combinations(b, t)))
    
    # Reverse index: t-subset -> list of block indices containing it
    rev_index = {}
    for bi, tsubs in enumerate(block_tsubs):
        for ts in tsubs:
            if ts not in rev_index:
                rev_index[ts] = []
            rev_index[ts].append(bi)
    
    result = {
        'all_blocks': all_blocks,
        'block_tsubs': block_tsubs,
        'rev_index': rev_index,
    }
    _CACHE[key] = result
    return result


def _greedy_construct(v, k, t, precomp, rng, deadline):
    """Randomized greedy: pick random uncovered t-subset, choose block covering most uncovered."""
    all_blocks = precomp['all_blocks']
    block_tsubs = precomp['block_tsubs']
    rev_index = precomp['rev_index']
    
    all_tsubs = list(rev_index.keys())
    uncovered = set(all_tsubs)
    selected = []
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset as pivot
        pivot = rng.choice(list(uncovered))
        candidates = rev_index.get(pivot, [])
        if not candidates:
            break
        
        best_block = None
        best_count = -1
        best_tie_break = None
        
        for bi in candidates:
            count = 0
            for ts in block_tsubs[bi]:
                if ts in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = bi
                best_tie_break = count
            elif count == best_count and count > 0:
                # Random tie-break
                if rng.random() < 0.5:
                    best_block = bi
        
        if best_block is None or best_count <= 0:
            break
        
        selected.append(best_block)
        # Update uncovered
        for ts in block_tsubs[best_block]:
            uncovered.discard(ts)
    
    return selected


def _deletion_phase(v, k, t, precomp, selected_blocks, deadline):
    """Remove redundant blocks: a block is redundant if all its t-subsets are covered by other selected blocks."""
    if not selected_blocks:
        return selected_blocks
    
    block_tsubs = precomp['block_tsubs']
    all_blocks = precomp['all_blocks']
    
    # Build coverage count: for each t-subset, how many selected blocks cover it
    coverage = Counter()
    for bi in selected_blocks:
        for ts in block_tsubs[bi]:
            coverage[ts] += 1
    
    changed = True
    current_selected = list(selected_blocks)
    
    while changed and time.time() < deadline:
        changed = False
        new_selected = []
        for bi in current_selected:
            # Check if this block is redundant
            redundant = True
            for ts in block_tsubs[bi]:
                if coverage[ts] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove this block: decrement coverage for its t-subsets
                for ts in block_tsubs[bi]:
                    coverage[ts] -= 1
                changed = True
            else:
                new_selected.append(bi)
        current_selected = new_selected
    
    return current_selected


def construct(v, k, t):
    """Construct a covering: list of k-subsets of {0,...,v-1} covering all t-subsets."""
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin before 150s kill
    
    precomp = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # First run: deterministic greedy (seed=0)
    rng = random.Random(0)
    selected = _greedy_construct(v, k, t, precomp, rng, deadline)
    cleaned = _deletion_phase(v, k, t, precomp, selected, deadline)
    if cleaned:
        best_blocks = cleaned
        best_count = len(cleaned)
    
    if time.time() >= deadline - 5:
        return [list(precomp['all_blocks'][bi]) for bi in best_blocks]
    
    # Multiple randomized restarts
    restart_num = 0
    while time.time() < deadline - 5:
        restart_num += 1
        seed = restart_num + 1000
        rng = random.Random(seed)
        
        selected = _greedy_construct(v, k, t, precomp, rng, deadline)
        if not selected:
            continue
        
        cleaned = _deletion_phase(v, k, t, precomp, selected, deadline)
        
        if cleaned and len(cleaned) < best_count:
            best_blocks = cleaned
            best_count = len(cleaned)
    
    return [list(precomp['all_blocks'][bi]) for bi in best_blocks]
