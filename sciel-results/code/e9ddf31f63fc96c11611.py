"""Covering design C(19,8,5) - optimized greedy with lazy count updates and repair."""
import time
import random
import itertools
from collections import defaultdict

_DEADLINE = 140  # safety margin before 150s kill


def construct(v, k, t):
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    # Precompute all t-subsets as bitmasks
    all_t_subsets = []
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_t_subsets.append(mask)
    
    n_t = len(all_t_subsets)  # C(19,5) = 11628
    
    # Build mask -> index lookup
    mask_to_idx = {}
    for i, mask in enumerate(all_t_subsets):
        mask_to_idx[mask] = i
    
    # Precompute all k-subsets
    all_k_subsets = []
    for combo in itertools.combinations(range(v), k):
        all_k_subsets.append(combo)
    
    n_blocks = len(all_k_subsets)  # C(19,8) = 75582
    
    # For each k-subset, compute the list of t-subset indices it covers
    # A t-subset is covered by a k-subset if all t points are in the k-subset
    block_covers = []
    for combo in all_k_subsets:
        covered = []
        for sub in itertools.combinations(combo, t):
            mask = 0
            for x in sub:
                mask |= (1 << x)
            idx = mask_to_idx.get(mask)
            if idx is not None:
                covered.append(idx)
        block_covers.append(covered)
    
    # Build reverse mapping: for each t-subset index, which blocks cover it
    t_covered_by = defaultdict(list)
    for bi, covered in enumerate(block_covers):
        for idx in covered:
            t_covered_by[idx].append(bi)
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks_used = []
        
        # Track how many uncovered t-subsets each block currently covers
        # Initialize by counting for each block how many of its covered t-subsets are in uncovered
        count_uncovered = [0] * n_blocks
        for bi in range(n_blocks):
            c = 0
            for idx in block_covers[bi]:
                if idx in uncovered:
                    c += 1
            count_uncovered[bi] = c
        
        while uncovered:
            if elapsed() > _DEADLINE:
                return None  # timeout
            
            # Find the block that covers the most uncovered t-subsets
            best_block = -1
            best_count = -1
            
            for bi in range(n_blocks):
                c = count_uncovered[bi]
                if c > best_count:
                    best_count = c
                    best_block = bi
                elif c == best_count and c > 0:
                    # Tie-break randomly for diversification
                    if rng.random() < 0.5:
                        best_block = bi
            
            if best_block == -1 or best_count == 0:
                # No block can cover any remaining uncovered subset
                break
            
            # Add this block
            blocks_used.append(list(all_k_subsets[best_block]))
            
            # Update counts: remove this block's covered t-subsets from uncovered
            # and decrement count_uncovered for all blocks that also cover those t-subsets
            for idx in block_covers[best_block]:
                if idx in uncovered:
                    uncovered.remove(idx)
                    # Decrement count for all blocks that cover this idx
                    for bj in t_covered_by[idx]:
                        count_uncovered[bj] -= 1
        
        return blocks_used
    
    # Try multiple random restarts
    best_blocks = None
    best_len = float('inf')
    
    for seed in range(300):
        if elapsed() > _DEADLINE:
            break
        
        result = greedy_cover(seed)
        if result is not None:
            if len(result) < best_len:
                best_len = len(result)
                best_blocks = result
        
        # Early termination if we hit a very good solution
        if best_len <= 270:
            break
    
    if best_blocks is None:
        # Fallback: simple greedy with seed 0
        best_blocks = greedy_cover(0)
        if best_blocks is None:
            # Last resort: return an empty list (shouldn't happen)
            return []
    
    # Verification pass: ensure every t-subset is covered
    # Build a set of all points in blocks for quick lookup
    # Actually, let's verify by checking each t-subset is contained in some block
    block_sets = [frozenset(b) for b in best_blocks]
    
    missing = []
    for tm in all_t_subsets:
        # Convert mask back to set of points
        points = set()
        for x in range(v):
            if tm & (1 << x):
                points.add(x)
        if not any(points.issubset(bs) for bs in block_sets):
            missing.append(tm)
    
    if missing:
        # Repair: add blocks to cover missing t-subsets
        for tm in missing:
            points = set()
            for x in range(v):
                if tm & (1 << x):
                    points.add(x)
            # Find a k-subset containing these points
            # Extend points to a k-subset
            remaining_points = set(range(v)) - points
            # Try to find a block that contains all missing points
            found = False
            for extra in itertools.combinations(sorted(remaining_points), k - len(points)):
                candidate = tuple(sorted(points | set(extra)))
                # Check if this block is already in best_blocks
                if candidate not in [tuple(sorted(b)) for b in best_blocks]:
                    best_blocks.append(list(candidate))
                    found = True
                    break
                # If it's already there, it should have covered this subset, so something's wrong
            if not found and elapsed() < _DEADLINE - 5:
                # Force add a block
                candidate = list(points)
                for p in sorted(remaining_points):
                    if len(candidate) >= k:
                        break
                    candidate.append(p)
                if len(candidate) > k:
                    candidate = candidate[:k]
                best_blocks.append(sorted(candidate))
    
    # Deduplicate
    seen = set()
    final_blocks = []
    for b in best_blocks:
        b_sorted = sorted(b)
        key = tuple(b_sorted)
        if key not in seen:
            seen.add(key)
            final_blocks.append(b_sorted)
    
    return final_blocks
