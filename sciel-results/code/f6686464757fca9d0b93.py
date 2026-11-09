import time
import itertools
import numpy as np
from collections import defaultdict

_SEED = 20240612

def _rng(seed):
    return np.random.RandomState(seed)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    rng = _rng(_SEED)

    # Precompute all triples and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    n_all = len(all_triples)
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}

    # Precompute pair-to-triple-index mapping for fast lookup
    # For each pair (a,b), store list of (triple_idx, c) where c is the third point
    pair_to_triples = defaultdict(list)
    for tr_idx, tr in enumerate(all_triples):
        for i in range(t):
            for j in range(i+1, t):
                pair = (tr[i], tr[j])
                c = tr[0] if i != 0 and j != 0 else (tr[1] if i != 1 and j != 1 else tr[2])
                # Find the point not in the pair
                for p in tr:
                    if p != tr[i] and p != tr[j]:
                        c = p
                        break
                pair_to_triples[pair].append((tr_idx, c))

    # Count of how many times each triple is covered
    count = np.zeros(n_all, dtype=np.int32)
    blocks = []
    block_set = set()

    def add_block(block):
        if block in block_set:
            return False
        block_set.add(block)
        blocks.append(block)
        for tr in itertools.combinations(block, t):
            count[triple_to_idx[tr]] += 1
        return True

    def remove_block(block):
        for tr in itertools.combinations(block, t):
            count[triple_to_idx[tr]] -= 1
        block_set.discard(block)

    def get_uncovered_indices():
        return np.where(count == 0)[0]

    def build_block_greedy(start_triple, count_arr, rng_obj):
        """Build a block greedily starting from a given triple. Optimized version."""
        block = list(start_triple)
        remaining = set(range(v))
        for p in start_triple:
            remaining.discard(p)

        for _ in range(k - t):
            best_cand = None
            best_s = -1
            # Get pairs from current block (t-1=2 for t=3)
            pairs = list(itertools.combinations(block, t-1))
            
            # Iterate candidates
            for cand in remaining:
                s = 0
                for p1, p2 in pairs:
                    # Use precomputed mapping for O(1) lookup
                    pair_key = (p1, p2) if p1 < p2 else (p2, p1)
                    for tr_idx, c in pair_to_triples[pair_key]:
                        if c == cand:
                            if count_arr[tr_idx] == 0:
                                s += 1
                            break
                if s > best_s:
                    best_s = s
                    best_cand = cand
            
            if best_cand is None:
                best_cand = next(iter(remaining))
            
            block.append(best_cand)
            remaining.discard(best_cand)
        
        return tuple(sorted(block))

    def compute_gain(block, count_arr):
        """Compute how many uncovered triples this block covers."""
        gain = 0
        for tr in itertools.combinations(block, t):
            if count_arr[triple_to_idx[tr]] == 0:
                gain += 1
        return gain

    # Phase 1: Greedy construction
    for _ in range(300):
        if time.time() > deadline - 10:
            break
        
        uncovered_indices = get_uncovered_indices()
        if len(uncovered_indices) == 0:
            break

        # Sample uncovered triples to find best starting point
        sample_size = min(30, len(uncovered_indices))
        sample_idx = rng.choice(uncovered_indices, size=sample_size, replace=False)

        best_block = None
        best_gain = -1

        for idx in sample_idx:
            tr = all_triples[idx]
            block = build_block_greedy(tr, count, rng)
            gain = compute_gain(block, count)
            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_block is None:
            break

        add_block(best_block)

    # Phase 2: Local improvement - try removing or replacing blocks
    for _ in range(150):
        if time.time() > deadline - 5:
            break
        improved = False
        for bi in range(len(blocks)):
            if time.time() > deadline - 5:
                break
            
            removed_block = blocks[bi]
            # Identify which triples will become uncovered if we remove this block
            critical_triples = []
            for tr in itertools.combinations(removed_block, t):
                if count[triple_to_idx[tr]] == 1:
                    critical_triples.append(tr)
            
            if not critical_triples:
                # Can remove this block safely
                remove_block(removed_block)
                blocks.pop(bi)
                improved = True
                break
            
            # Try to find a replacement block that covers all critical triples
            best_repl = None
            best_score = -1
            
            # Strategy 1: Build from each critical triple
            for start_tr in critical_triples:
                block = build_block_greedy(start_tr, count, rng)
                
                # Check if this block covers all critical_triples
                block_triples = set(itertools.combinations(block, t))
                valid = all(tr in block_triples for tr in critical_triples)
                
                if valid:
                    gain = compute_gain(block, count)
                    if gain > best_score:
                        best_score = gain
                        best_repl = block
            
            # Strategy 2: Try to build a block that maximally overlaps with critical triples
            # Find points that appear in multiple critical triples
            point_freq = defaultdict(int)
            for tr in critical_triples:
                for p in tr:
                    point_freq[p] += 1
            
            # Sort points by frequency
            sorted_points = sorted(point_freq.keys(), key=lambda p: -point_freq[p])
            
            # Try building from top frequent points
            for p in sorted_points[:5]:
                # Find critical triples containing p
                relevant_triples = [tr for tr in critical_triples if p in tr]
                if not relevant_triples:
                    continue
                
                # Start with a triple containing p
                start_tr = relevant_triples[0]
                block = build_block_greedy(start_tr, count, rng)
                
                block_triples = set(itertools.combinations(block, t))
                valid = all(tr in block_triples for tr in critical_triples)
                
                if valid:
                    gain = compute_gain(block, count)
                    if gain > best_score:
                        best_score = gain
                        best_repl = block
            
            if best_repl is not None:
                remove_block(removed_block)
                blocks[bi] = best_repl
                add_block(best_repl)
                improved = True
                break

        if not improved:
            break

    # Phase 3: Ensure full coverage
    while time.time() < deadline - 2:
        uncovered_indices = get_uncovered_indices()
        if len(uncovered_indices) == 0:
            break
        tr = all_triples[uncovered_indices[0]]
        block = build_block_greedy(tr, count, rng)
        add_block(block)

    # Final deduplication (should be empty due to block_set, but safety check)
    seen = set()
    unique_blocks = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            unique_blocks.append(b)

    return [list(b) for b in unique_blocks]
