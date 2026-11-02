import numpy as np
import itertools
import time
import random
import heapq
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    # Parameters for C(24, 7, 3)
    # v=24, k=7, t=3
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute block -> subsets mapping
    # block_subsets[i] is a list of indices of t-subsets covered by block i
    block_subsets = []
    for i, block in enumerate(all_ks):
        subs = [idx_map[sub] for sub in itertools.combinations(block, t)]
        block_subsets.append(subs)
        
    # Precompute subset -> blocks mapping
    # subset_blocks[j] is a list of indices of k-blocks covering subset j
    subset_blocks = [[] for _ in range(n_ts)]
    for b_idx in range(n_ks):
        for s_idx in block_subsets[b_idx]:
            subset_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for faster access if needed, but lists of ints are often faster for small sets
    # Let's keep them as lists.
    
    def run_lazy_greedy(seed):
        random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        remaining = n_ts
        blocks_indices = []
        
        # Marginal gain for each block: number of uncovered subsets it covers
        # Initially, gain[i] = len(block_subsets[i])
        # We use a heap of (-gain, block_idx)
        # Lazy evaluation: we push new gains, and when popping, we check if the gain is still valid.
        
        # To avoid recomputing gain for all blocks initially, we note that gain[i] is constant initially.
        # But after covering some subsets, gain changes.
        
        # Efficient Lazy Greedy:
        # 1. Heap of (-gain, idx)
        # 2. current_gain[i] array
        # 3. When we pick a block, we update 'covered'.
        # 4. For every block that intersects with the newly covered subsets, we recompute its gain and push to heap.
        
        # Which blocks intersect?
        # We can maintain a set of "active" blocks or just iterate through all blocks if n_ks is small?
        # n_ks = 346104. Iterating all is too slow if done many times.
        # We need to know which blocks cover the newly covered subsets.
        
        # Let's maintain:
        # cover_count[s] : number of blocks in solution covering subset s (for deletion check later)
        # But for greedy, we just need 'covered' bool array.
        
        # To find blocks to update:
        # When we add block B, it covers subsets S_B.
        # Any block B' that contains any s in S_B needs its gain updated.
        # We can iterate through s in S_B, and for each s, iterate through blocks in subset_blocks[s].
        # This is efficient if S_B is small (it is, size 35) and subset_blocks[s] is small (size 280).
        # 35 * 280 = 9800 updates per block added.
        # Number of blocks added ~ 70.
        # Total updates ~ 700,000. This is very fast.
        
        current_gain = np.array([len(subs) for subs in block_subsets], dtype=np.int32)
        heap = []
        # Initialize heap
        for i in range(n_ks):
            heapq.heappush(heap, (-current_gain[i], i))
            
        # We need to track which blocks are already in the solution to avoid re-adding?
        # No, if we pop a block already in solution, its gain would be 0 (since all its subsets are covered).
        # Actually, if a block is in the solution, its subsets are covered.
        # If we update its gain, it will be 0.
        # So we can just skip if gain is 0.
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
                
            # Pop from heap
            neg_gain, b_idx = heapq.heappop(heap)
            gain = -neg_gain
            
            # Check if this gain is still valid
            # If current_gain[b_idx] != gain, it's stale.
            # Note: current_gain only decreases.
            if gain != current_gain[b_idx]:
                continue
                
            if gain == 0:
                # No more blocks can cover new subsets
                break
                
            # Add this block
            blocks_indices.append(b_idx)
            
            # Update covered and current_gain for intersecting blocks
            subs_in_block = block_subsets[b_idx]
            for s_idx in subs_in_block:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    remaining -= 1
                    
            # Update gains for all blocks that cover any of these subsets
            # To avoid duplicate updates, we can use a set or just push duplicates (lazy heap handles it)
            blocks_to_update = set()
            for s_idx in subs_in_block:
                for b_prime in subset_blocks[s_idx]:
                    blocks_to_update.add(b_prime)
                    
            for b_prime in blocks_to_update:
                # Recompute gain for b_prime
                # gain = count of s in block_subsets[b_prime] where covered[s] is False
                new_gain = 0
                for s_idx in block_subsets[b_prime]:
                    if not covered[s_idx]:
                        new_gain += 1
                if new_gain != current_gain[b_prime]:
                    current_gain[b_prime] = new_gain
                    heapq.heappush(heap, (-new_gain, b_prime))
                    
        return blocks_indices, len(blocks_indices)

    def reduce_solution(blocks_indices):
        """
        Remove redundant blocks.
        A block is redundant if all its subsets are covered by other blocks.
        """
        if not blocks_indices:
            return blocks_indices
            
        # Count coverage
        cover_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in blocks_indices:
            for s_idx in block_subsets[b_idx]:
                cover_count[s_idx] += 1
                
        kept = []
        for b_idx in blocks_indices:
            # Check if any subset in this block has cover_count == 1
            # If yes, it's critical, keep it.
            # If no, it's redundant, remove it.
            is_critical = False
            for s_idx in block_subsets[b_idx]:
                if cover_count[s_idx] == 1:
                    is_critical = True
                    break
                    
            if is_critical:
                kept.append(b_idx)
            else:
                # Remove it
                for s_idx in block_subsets[b_idx]:
                    cover_count[s_idx] -= 1
                    
        if len(kept) < len(blocks_indices):
            # Recurse or loop until stable
            return reduce_solution(kept)
        else:
            return kept

    def local_search_improve(blocks_indices):
        """
        Improve by swapping blocks to enable further reduction.
        """
        current_blocks = list(blocks_indices)
        current_set = set(current_blocks)
        
        # Build cover count
        cover_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subsets[b_idx]:
                cover_count[s_idx] += 1
                
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
                
            improved = False
            
            # First, reduce
            reduced = reduce_solution(current_blocks)
            if len(reduced) < len(current_blocks):
                current_blocks = reduced
                current_set = set(current_blocks)
                # Rebuild cover count
                cover_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in current_blocks:
                    for s_idx in block_subsets[b_idx]:
                        cover_count[s_idx] += 1
                improved = True
                continue
            
            # Try swaps
            # For each block B, find its unique subsets U_B
            # If U_B is not empty, try to find B' covering U_B
            # If found, replace B with B'
            
            # To make this efficient, we iterate and check.
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                
                # Find unique subsets for b_idx
                unique_subs = []
                for s_idx in block_subsets[b_idx]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                        
                if not unique_subs:
                    # This block is redundant? 
                    # reduce_solution should have caught this. 
                    # But maybe cover_count is stale? 
                    # If we just reduced, cover_count is fresh.
                    # So this shouldn't happen unless reduce_solution failed?
                    # Let's just remove it to be safe.
                    current_blocks.pop(i)
                    current_set.discard(b_idx)
                    for s_idx in block_subsets[b_idx]:
                        cover_count[s_idx] -= 1
                    improved = True
                    break
                    
                # Try to find a replacement
                # Intersection of blocks covering all s in unique_subs
                # Start with blocks covering the first unique sub
                candidates = set(subset_blocks[unique_subs[0]])
                
                for s_idx in unique_subs[1:]:
                    candidates.intersection_update(subset_blocks[s_idx])
                    if not candidates:
                        break
                        
                if candidates:
                    # Remove current block from candidates
                    candidates.discard(b_idx)
                    
                    if candidates:
                        # Pick a candidate
                        # Heuristic: Pick one that covers many other subsets?
                        # Or just random?
                        # Let's pick the one with highest "potential" (covers many subsets with count 1? No, that would break them).
                        # Just pick the first one.
                        new_b = next(iter(candidates))
                        
                        # Replace
                        current_blocks[i] = new_b
                        current_set.discard(b_idx)
                        current_set.add(new_b)
                        
                        # Update cover counts
                        for s_idx in block_subsets[b_idx]:
                            cover_count[s_idx] -= 1
                        for s_idx in block_subsets[new_b]:
                            cover_count[s_idx] += 1
                            
                        improved = True
                        break
                        
        return current_blocks

    best_solution = None
    best_score = float('inf')
    
    # Run multiple instances
    restarts = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restarts
        blocks, score = run_lazy_greedy(seed)
        
        if blocks is None:
            break
            
        # Post-process
        blocks = reduce_solution(blocks)
        blocks = local_search_improve(blocks)
        blocks = reduce_solution(blocks)
        
        if len(blocks) < best_score:
            best_score = len(blocks)
            best_solution = blocks
            
        restarts += 1
        
        # If we get a very good score, we can stop or continue
        if best_score <= 70:
            # Try a few more to see if we can get 69
            if restarts > 20:
                break

    if best_solution is None:
        # Fallback: run one greedy if time permits
        blocks, _ = run_lazy_greedy(0)
        if blocks:
            blocks = reduce_solution(blocks)
            best_solution = blocks
        else:
            best_solution = []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
