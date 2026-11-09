"""
Covering design C(v, k, t) via Johnson-scheme-inspired probabilistic construction
with targeted local search repair.

Strategy:
1. Build a random symmetric matrix to approximate a Steiner system structure.
2. Use the matrix to guide block selection (blocks with high "structural" scores).
3. Greedily add blocks to cover all t-subsets.
4. Local search: remove redundant blocks, then replace blocks to minimize overlap
   while maintaining coverage of uniquely-covered t-subsets.
"""
import itertools
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1).
    """
    # Generate all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Coverage array
    covered = np.zeros(n_tsubs, dtype=np.int32)
    
    def get_tsubs_of_block(block):
        """Get indices of all t-subsets contained in a block."""
        idxs = []
        for ts in itertools.combinations(sorted(block), t):
            idxs.append(tsub_to_idx[ts])
        return idxs
    
    def add_block(block):
        for idx in get_tsubs_of_block(block):
            covered[idx] += 1
    
    def remove_block(block):
        for idx in get_tsubs_of_block(block):
            covered[idx] -= 1
    
    def uncovered_count():
        return int(np.sum(covered == 0))
    
    def block_gain(block, uncovered_set):
        """Count how many uncovered t-subsets this block covers."""
        gain = 0
        for ts in itertools.combinations(sorted(block), t):
            idx = tsub_to_idx[ts]
            if idx in uncovered_set:
                gain += 1
        return gain
    
    # Phase 0: Johnson-scheme-inspired structure via random symmetric matrix
    # The idea: a Steiner system S(t,k,v) has the property that every t-subset
    # is in exactly one block. We approximate this by building a random symmetric
    # matrix M where M[i,j] encodes "affinity" between points i and j.
    # Blocks are selected to have high internal affinity (sum of M[i,j] for i,j in block).
    
    rng = np.random.RandomState(42)
    
    # Build a random symmetric affinity matrix with zero diagonal
    # Use a low-rank structure to encourage clustering
    rank = max(2, v // 6)
    A = rng.randn(v, rank)
    M = A @ A.T  # v x v symmetric PSD matrix
    np.fill_diagonal(M, 0)
    
    # Normalize to [0, 1] for interpretability
    M_min, M_max = M.min(), M.max()
    if M_max > M_min:
        M = (M - M_min) / (M_max - M_min)
    else:
        M = np.zeros_like(M)
    
    def block_affinity(block):
        """Sum of pairwise affinities within a block."""
        b = list(block)
        aff = 0.0
        for i in range(len(b)):
            for j in range(i+1, len(b)):
                aff += M[b[i], b[j]]
        return aff
    
    # Phase 1: Initial greedy construction with structure-guided sampling
    blocks = []
    used_blocks = set()
    uncovered_set = set(range(n_tsubs))
    
    # Precompute all possible blocks? No, too many. Sample instead.
    # But we want to bias toward high-affinity blocks.
    
    # First, generate a pool of candidate blocks biased by affinity
    n_candidates = 500
    candidate_blocks = []
    for _ in range(n_candidates):
        # Sample a block with bias toward high-affinity subsets
        # Use a simple approach: sample random blocks, keep those with above-median affinity
        block = sorted(random.sample(range(v), k))
        aff = block_affinity(block)
        candidate_blocks.append((aff, tuple(block)))
    
    # Sort by affinity descending
    candidate_blocks.sort(key=lambda x: -x[0])
    
    # Also add some purely random blocks for diversity
    for _ in range(200):
        block = sorted(random.sample(range(v), k))
        candidate_blocks.append((0.0, tuple(block)))
    
    # Greedy phase: repeatedly pick the block with highest gain among candidates
    # Re-sample candidates when gain drops
    max_greedy_blocks = 150
    for _ in range(max_greedy_blocks):
        if not uncovered_set:
            break
        
        best_gain = 0
        best_block = None
        
        # Evaluate top candidates by gain
        for aff, blk in candidate_blocks[:300]:
            if blk in used_blocks:
                continue
            gain = block_gain(blk, uncovered_set)
            if gain > best_gain:
                best_gain = gain
                best_block = blk
        
        if best_block is None or best_gain == 0:
            # No good candidate found; generate new ones
            new_candidates = []
            for _ in range(300):
                block = sorted(random.sample(range(v), k))
                aff = block_affinity(block)
                new_candidates.append((aff, tuple(block)))
            new_candidates.sort(key=lambda x: -x[0])
            candidate_blocks = new_candidates
            # Retry once
            for aff, blk in candidate_blocks[:200]:
                if blk in used_blocks:
                    continue
                gain = block_gain(blk, uncovered_set)
                if gain > best_gain:
                    best_gain = gain
                    best_block = blk
            
            if best_block is None or best_gain == 0:
                # Targeted: pick an uncovered t-subset and build a block around it
                idx = next(iter(uncovered_set))
                ts = all_tsubs[idx]
                base = list(ts)
                remaining = [x for x in range(v) if x not in set(base)]
                random.shuffle(remaining)
                block = tuple(sorted(base + remaining[:k-t]))
                if block not in used_blocks:
                    best_block = block
                    best_gain = 1
                else:
                    # Try different extensions
                    found = False
                    for _ in range(20):
                        random.shuffle(remaining)
                        block = tuple(sorted(base + remaining[:k-t]))
                        if block not in used_blocks:
                            best_block = block
                            best_gain = 1
                            found = True
                            break
                    if not found:
                        break
        
        if best_block is not None:
            blocks.append(best_block)
            used_blocks.add(best_block)
            add_block(list(best_block))
            # Update uncovered set
            for idx in get_tsubs_of_block(list(best_block)):
                if covered[idx] == 1:
                    uncovered_set.discard(idx)
    
    # Phase 2: Continue until all covered
    while uncovered_set:
        idx = next(iter(uncovered_set))
        ts = all_tsubs[idx]
        base = list(ts)
        remaining = [x for x in range(v) if x not in set(base)]
        random.shuffle(remaining)
        block = tuple(sorted(base + remaining[:k-t]))
        
        # Try to find a block that covers this and maximizes additional coverage
        best_block = block
        best_gain = 1
        for _ in range(50):
            random.shuffle(remaining)
            cand = tuple(sorted(base + remaining[:k-t]))
            if cand in used_blocks:
                continue
            gain = block_gain(cand, uncovered_set)
            if gain > best_gain:
                best_gain = gain
                best_block = cand
        
        blocks.append(best_block)
        used_blocks.add(best_block)
        add_block(list(best_block))
        for idx2 in get_tsubs_of_block(list(best_block)):
            if covered[idx2] == 1:
                uncovered_set.discard(idx2)
    
    # Phase 3: Local search to reduce block count
    # 3a: Remove redundant blocks
    changed = True
    while changed and len(blocks) > 40:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            block = list(blocks[i])
            remove_block(block)
            if uncovered_count() == 0:
                blocks.pop(i)
                used_blocks.discard(tuple(block))
                changed = True
                break
            else:
                add_block(block)
    
    # 3b: Replace blocks to reduce overlap
    for iteration in range(30):
        improved = False
        for i in range(len(blocks)):
            old_block = list(blocks[i])
            old_tuple = tuple(sorted(old_block))
            
            # Find t-subsets only covered by this block
            remove_block(old_block)
            uniquely_covered = []
            for idx in get_tsubs_of_block(old_block):
                if covered[idx] == 0:
                    uniquely_covered.append(idx)
            add_block(old_block)
            
            if not uniquely_covered:
                # Redundant; try to remove
                remove_block(old_block)
                if uncovered_count() == 0:
                    blocks.pop(i)
                    used_blocks.discard(old_tuple)
                    improved = True
                    break
                add_block(old_block)
                continue
            
            # Get required points
            required_points = set()
            for idx in uniquely_covered:
                required_points.update(all_tsubs[idx])
            
            if len(required_points) > k:
                continue
            
            # Try to find a replacement with less overlap
            best_new_block = None
            best_overlap = float('inf')
            
            for _ in range(30):
                new_block = list(required_points)
                remaining = [x for x in range(v) if x not in set(new_block)]
                random.shuffle(remaining)
                new_block += remaining[:k - len(new_block)]
                new_tuple = tuple(sorted(new_block))
                
                if new_tuple == old_tuple:
                    continue
                if new_tuple in used_blocks:
                    continue
                
                # Check coverage of uniquely_covered
                new_set = set(new_block)
                covers_all = all(set(all_tsubs[idx]).issubset(new_set) for idx in uniquely_covered)
                if not covers_all:
                    continue
                
                # Calculate overlap with other blocks
                remove_block(old_block)
                overlap = 0
                for idx in get_tsubs_of_block(list(new_block)):
                    if covered[idx] > 0:
                        overlap += 1
                add_block(old_block)
                
                if overlap < best_overlap:
                    best_overlap = overlap
                    best_new_block = new_tuple
            
            if best_new_block is not None:
                remove_block(old_block)
                blocks[i] = best_new_block
                used_blocks.discard(old_tuple)
                used_blocks.add(best_new_block)
                add_block(list(best_new_block))
                improved = True
        
        if not improved:
            break
    
    # Final cleanup: remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            block = list(blocks[i])
            remove_block(block)
            if uncovered_count() == 0:
                blocks.pop(i)
                used_blocks.discard(tuple(block))
                changed = True
                break
            else:
                add_block(block)
    
    # Ensure all t-subsets are covered
    while uncovered_count() > 0:
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break
        idx = uncovered_indices[0]
        ts = all_tsubs[idx]
        base = list(ts)
        remaining = [x for x in range(v) if x not in set(base)]
        random.shuffle(remaining)
        block = tuple(sorted(base + remaining[:k-t]))
        
        # Try to find a non-duplicate block
        attempts = 0
        while block in used_blocks and attempts < 50:
            random.shuffle(remaining)
            block = tuple(sorted(base + remaining[:k-t]))
            attempts += 1
        
        if block not in used_blocks:
            blocks.append(block)
            used_blocks.add(block)
            add_block(list(block))
        else:
            # Force add even if duplicate (shouldn't happen often)
            blocks.append(block)
            add_block(list(block))
    
    # Deduplicate just in case
    seen = set()
    final_blocks = []
    for b in blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            final_blocks.append(list(b))
    
    return final_blocks
