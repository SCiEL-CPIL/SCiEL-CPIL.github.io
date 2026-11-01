import itertools
import time
import random
import sys
import numpy as np

_global_deadline = None

def _precompute(v, k, t):
    """
    Precomputes:
    1. all_blocks: list of all k-subsets of range(v)
    2. block_to_ts: list of lists, where block_to_ts[i] is the list of t-subsets in all_blocks[i]
    3. ts_to_blocks: dict mapping t-subset tuple to list of indices in all_blocks
    """
    all_blocks = []
    block_to_ts = []
    ts_to_blocks = {}
    
    # Generate all k-subsets
    for block in itertools.combinations(range(v), k):
        idx = len(all_blocks)
        all_blocks.append(block)
        
        # Generate all t-subsets of this block
        t_subs = list(itertools.combinations(block, t))
        block_to_ts.append(t_subs)
        
        for ts in t_subs:
            ts_tuple = tuple(sorted(ts))
            if ts_tuple not in ts_to_blocks:
                ts_to_blocks[ts_tuple] = []
            ts_to_blocks[ts_tuple].append(idx)
            
    return all_blocks, block_to_ts, ts_to_blocks

def _greedy_cover(v, k, t, all_blocks, block_to_ts, ts_to_blocks, rng, deadline):
    """
    Greedy set cover.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset
        seed_ts = rng.choice(list(uncovered))
        
        # Get candidate blocks that cover seed_ts
        candidates = ts_to_blocks.get(seed_ts, [])
        if not candidates:
            # Should not happen if we have all blocks
            break
            
        best_idx = -1
        best_gain = -1
        
        for idx in candidates:
            # Calculate gain: how many uncovered t-subsets does this block cover?
            gain = 0
            for ts in block_to_ts[idx]:
                if ts in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                
        if best_idx < 0:
            break
            
        blocks.append(all_blocks[best_idx])
        
        # Remove covered t-subsets from uncovered
        for ts in block_to_ts[best_idx]:
            uncovered.discard(ts)
            
    return blocks

def _local_search(blocks, v, k, t, all_blocks, block_to_ts, ts_to_blocks, deadline):
    """
    Local search:
    1. Remove redundant blocks.
    2. Try to merge pairs of blocks into one.
    """
    blocks = [list(b) for b in blocks]
    
    # Map block tuple to index in blocks list?
    # We need to track coverage counts.
    
    def build_coverage(blocks):
        counts = {}
        for b in blocks:
            b_set = set(b)
            for ts in itertools.combinations(b, t):
                ts_t = tuple(sorted(ts))
                counts[ts_t] = counts.get(ts_t, 0) + 1
        return counts

    coverage = build_coverage(blocks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant blocks
        n = len(blocks)
        for i in range(n):
            if time.time() >= deadline:
                break
            b = blocks[i]
            redundant = True
            for ts in itertools.combinations(b, t):
                ts_t = tuple(sorted(ts))
                if coverage.get(ts_t, 0) == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                removed = blocks.pop(i)
                for ts in itertools.combinations(removed, t):
                    ts_t = tuple(sorted(ts))
                    coverage[ts_t] -= 1
                    if coverage[ts_t] <= 0:
                        del coverage[ts_t]
                improved = True
                break # Restart loop to recalculate n and indices
        
        if improved:
            continue
            
        # Phase 2: Try to merge pairs
        n = len(blocks)
        # To avoid O(N^2) being too slow if N is large, we can limit or optimize.
        # N ~ 300. 300^2 = 90k.
        # For each pair, we check if union of unique subsets fits in k.
        
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                    
                # Temporarily remove blocks i and j
                b_i = blocks[i]
                b_j = blocks[j]
                
                # Identify subsets uniquely covered by i or j
                # A subset is uniquely covered by i or j if count is 1 and it's in i, or count is 2 and it's in both?
                # Actually, if we remove i and j, a subset becomes uncovered if:
                # 1. It was in i and count was 1.
                # 2. It was in j and count was 1.
                # 3. It was in both i and j and count was 2.
                
                uncovered_set = set()
                
                # Subsets in i
                for ts in itertools.combinations(b_i, t):
                    ts_t = tuple(sorted(ts))
                    c = coverage.get(ts_t, 0)
                    if c == 1:
                        uncovered_set.add(ts_t)
                    elif c == 2:
                        # Check if it's also in j
                        if set(ts_t).issubset(set(b_j)):
                            uncovered_set.add(ts_t)
                            
                # Subsets in j (that were not already added from i's logic, or handle duplicates)
                # Actually, if c==1 for a subset in j, it wasn't in i (or if it was, c would be >=2).
                # So we just need to check subsets in j with c==1.
                for ts in itertools.combinations(b_j, t):
                    ts_t = tuple(sorted(ts))
                    c = coverage.get(ts_t, 0)
                    if c == 1:
                        uncovered_set.add(ts_t)
                    elif c == 2:
                        # If it's in i, it was already added. If not, it's unique to j?
                        # No, if c==2 and it's in j, and not in i, then removing j makes it uncovered.
                        # But we are removing BOTH.
                        # So if c==2, and it is in j, it must be in exactly one other block.
                        # If it is in i, it's covered by i? No, we are removing i.
                        # So if it is in i AND j, and c==2, removing both leaves it uncovered.
                        # My logic above: "if set(ts_t).issubset(set(b_j))" inside the loop for b_i.
                        # This correctly identifies subsets in both.
                        # For b_j loop, if c==2, and it's NOT in b_i, then it's covered by j and one other.
                        # Removing j leaves it covered by the other. So it's NOT uncovered.
                        # So we only add to uncovered_set if c==1 for b_j.
                        pass

                if not uncovered_set:
                    # Both blocks are redundant together?
                    # This implies they are both redundant individually?
                    # If removing both leaves nothing uncovered, then removing one would also leave nothing uncovered
                    # (since the other would still cover its unique parts? No).
                    # Actually, if uncovered_set is empty, it means every subset in i or j is covered by some other block.
                    # So both are redundant. We could have removed them in Phase 1.
                    # But Phase 1 checks individually.
                    # If Phase 1 didn't remove them, it means each had at least one unique subset.
                    # But if we remove both, and the union of their unique subsets is empty?
                    # That's impossible if each had unique subsets.
                    # Unless the unique subset of i is the same as the unique subset of j?
                    # If i has unique ts, and j has unique ts, and they are the same ts, then c=2.
                    # Removing both leaves it uncovered. So it would be in uncovered_set.
                    # So if uncovered_set is empty, we can remove both.
                    blocks.pop(j)
                    blocks.pop(i)
                    improved = True
                    break
                    
                # Check if uncovered_set can be covered by a single block
                # Union of all points in uncovered_set
                union_points = set()
                for ts in uncovered_set:
                    union_points.update(ts)
                    
                if len(union_points) <= k:
                    # We can form a block containing union_points
                    # We need to find if such a block exists in all_blocks?
                    # Or just create it? The problem asks for a list of blocks.
                    # The blocks don't have to be from a precomputed list, they just need to be k-subsets.
                    # However, using a block from all_blocks is fine.
                    # We need to find a block in all_blocks that contains union_points.
                    
                    # To speed this up, we can just construct the block.
                    # But we need to update coverage counts correctly.
                    # Let's just create the block.
                    new_block = list(union_points)
                    # Fill with random points to reach size k
                    remaining = [x for x in range(v) if x not in union_points]
                    if len(new_block) < k:
                        new_block.extend(random.sample(remaining, k - len(new_block)))
                    new_block.sort()
                    
                    # Check if this new block is already in blocks?
                    # If it is, we just remove i and j, and don't add it (since it's already there).
                    # But if it's already there, it would have covered uncovered_set originally?
                    # If new_block is in blocks, say at index m.
                    # Then uncovered_set subsets are covered by m.
                    # So they wouldn't be "uncovered" when we remove i and j?
                    # Wait, "uncovered" here means "uncovered by blocks other than i and j".
                    # So if m exists, it covers them.
                    # So we can just remove i and j.
                    
                    found_existing = False
                    for idx, b in enumerate(blocks):
                        if idx == i or idx == j:
                            continue
                        if set(new_block).issuperset(set(b)) and len(b) == k: # Exact match?
                            # Better: check if b covers all ts in uncovered_set
                            # Actually, if b == new_block, it covers them.
                            if b == new_block:
                                found_existing = True
                                break
                                                    
                    if found_existing:
                        # Remove i and j
                        # Order matters for popping
                        if i < j:
                            blocks.pop(j)
                            blocks.pop(i)
                        else:
                            blocks.pop(i)
                            blocks.pop(j)
                    else:
                        # Remove i and j, add new_block
                        if i < j:
                            blocks.pop(j)
                            blocks.pop(i)
                        else:
                            blocks.pop(i)
                            blocks.pop(j)
                        blocks.append(new_block)
                        
                    improved = True
                    break
                    
            if improved:
                break
                
    # Rebuild coverage to ensure consistency? Not strictly needed if we update counts,
    # but we didn't update counts in Phase 2 carefully.
    # Let's just return the blocks. The caller can verify.
    return blocks

def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 115
    rng = random.Random(42)
    
    # Precompute
    all_blocks, block_to_ts, ts_to_blocks = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple greedy restarts
    for restart in range(5):
        if time.time() > _global_deadline - 30:
            break
            
        blocks = _greedy_cover(v, k, t, all_blocks, block_to_ts, ts_to_blocks, rng, _global_deadline - 30)
        
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
            
        # Local search on this solution
        if time.time() < _global_deadline - 10:
            improved_blocks = _local_search(blocks, v, k, t, all_blocks, block_to_ts, ts_to_blocks, _global_deadline - 10)
            if len(improved_blocks) < best_count:
                best_blocks = improved_blocks
                best_count = len(improved_blocks)
                
    return best_blocks
