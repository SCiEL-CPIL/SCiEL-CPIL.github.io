import itertools
import time
import random
import heapq
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110  # safety margin
    
    # Precompute all blocks and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, compute its t-subsets
    block_to_ts = []
    for block in all_blocks:
        t_subs = list(itertools.combinations(block, t))
        block_to_ts.append(t_subs)
    
    # Build reverse index: t-subset -> list of block indices
    ts_to_blocks = {}
    for i, t_subs in enumerate(block_to_ts):
        for ts in t_subs:
            if ts not in ts_to_blocks:
                ts_to_blocks[ts] = []
            ts_to_blocks[ts].append(i)
    
    # Number of t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    num_ts = len(all_ts)
    
    # Convert block_to_ts to use indices for faster lookup
    block_to_ts_idx = []
    for t_subs in block_to_ts:
        block_to_ts_idx.append([ts_index[ts] for ts in t_subs])
    
    def greedy_with_heap(uncovered_mask, block_gains, blocks_used, deadline):
        """
        Greedy set cover using max-heap on gains.
        uncovered_mask: numpy boolean array of length num_ts
        block_gains: numpy array of length num_blocks
        blocks_used: list of block indices used so far
        """
        # Build initial heap
        heap = []
        for i in range(num_blocks):
            if block_gains[i] > 0:
                heapq.heappush(heap, (-block_gains[i], i))
        
        while uncovered_mask.any() and time.time() < deadline:
            # Pop max gain block
            while heap:
                neg_gain, idx = heapq.heappop(heap)
                if block_gains[idx] == -neg_gain and block_gains[idx] > 0:
                    break
            else:
                break
            
            # Add this block
            blocks_used.append(idx)
            
            # Update uncovered mask and gains
            for ts_idx in block_to_ts_idx[idx]:
                if uncovered_mask[ts_idx]:
                    uncovered_mask[ts_idx] = False
                    # Decrement gain of all blocks containing this t-subset
                    for other_idx in ts_to_blocks[all_ts[ts_idx]]:
                        if other_idx != idx:
                            block_gains[other_idx] -= 1
                            if block_gains[other_idx] > 0:
                                heapq.heappush(heap, (-block_gains[other_idx], other_idx))
        
        return blocks_used
    
    def local_search(blocks, deadline):
        """
        Local search: remove redundant blocks, try swaps.
        """
        blocks = list(blocks)
        
        def compute_coverage(block_list):
            cov = np.zeros(num_ts, dtype=np.int32)
            for bi in block_list:
                for ts_idx in block_to_ts_idx[bi]:
                    cov[ts_idx] += 1
            return cov
        
        cov = compute_coverage(blocks)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Remove redundant blocks
            n = len(blocks)
            for i in range(n):
                if time.time() >= deadline:
                    break
                bi = blocks[i]
                redundant = True
                for ts_idx in block_to_ts_idx[bi]:
                    if cov[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in block_to_ts_idx[bi]:
                        cov[ts_idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try swaps
            n = len(blocks)
            for i in range(n):
                if time.time() >= deadline:
                    break
                bi = blocks[i]
                # Find t-subsets uniquely covered by bi
                unique_ts = [ts_idx for ts_idx in block_to_ts_idx[bi] if cov[ts_idx] == 1]
                if not unique_ts:
                    continue
                
                # Find candidate blocks that cover all unique_ts
                candidates = None
                for ts_idx in unique_ts:
                    ts = all_ts[ts_idx]
                    cand_set = set(ts_to_blocks[ts])
                    if candidates is None:
                        candidates = cand_set
                    else:
                        candidates &= cand_set
                
                if not candidates:
                    continue
                
                candidates.discard(bi)
                existing = set(blocks)
                candidates -= existing
                
                best_swap = None
                best_gain = -1
                for cand in candidates:
                    gain = 0
                    for ts_idx in block_to_ts_idx[cand]:
                        if cov[ts_idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = cand
                
                if best_swap is not None and best_gain > 0:
                    # Remove bi
                    for ts_idx in block_to_ts_idx[bi]:
                        cov[ts_idx] -= 1
                    # Add best_swap
                    for ts_idx in block_to_ts_idx[best_swap]:
                        cov[ts_idx] += 1
                    blocks[i] = best_swap
                    improved = True
                    break
        
        return blocks
    
    best_blocks = None
    best_count = float('inf')
    
    # Strategy 1: Deterministic greedy (rarest first)
    if time.time() < deadline - 30:
        uncovered_mask = np.ones(num_ts, dtype=bool)
        block_gains = np.array([len(block_to_ts_idx[i]) for i in range(num_blocks)], dtype=np.int32)
        blocks_used = []
        
        # Rarest-first: at each step, find uncovered t-subset with fewest covering blocks
        # Then pick the block among those that has highest gain
        while uncovered_mask.any() and time.time() < deadline - 30:
            # Find uncovered t-subset with minimum number of candidate blocks
            min_count = float('inf')
            seed_ts_idx = -1
            for ts_idx in range(num_ts):
                if not uncovered_mask[ts_idx]:
                    continue
                count = len(ts_to_blocks[all_ts[ts_idx]])
                if count < min_count:
                    min_count = count
                    seed_ts_idx = ts_idx
                    if min_count == 1:
                        break
            
            if seed_ts_idx < 0:
                break
            
            ts = all_ts[seed_ts_idx]
            candidates = ts_to_blocks[ts]
            
            best_idx = -1
            best_gain = -1
            for idx in candidates:
                gain = 0
                for ts_idx in block_to_ts_idx[idx]:
                    if uncovered_mask[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx < 0:
                break
            
            blocks_used.append(best_idx)
            for ts_idx in block_to_ts_idx[best_idx]:
                uncovered_mask[ts_idx] = False
        
        if len(blocks_used) < best_count:
            best_blocks = blocks_used
            best_count = len(blocks_used)
        
        # Local search
        if time.time() < deadline - 10:
            improved = local_search(blocks_used, deadline - 10)
            if len(improved) < best_count:
                best_blocks = improved
                best_count = len(improved)
    
    # Strategy 2: Random greedy with different seeds
    for seed in [42, 123, 456, 789, 1000, 2024]:
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(seed)
        uncovered_mask = np.ones(num_ts, dtype=bool)
        block_gains = np.array([len(block_to_ts_idx[i]) for i in range(num_blocks)], dtype=np.int32)
        blocks_used = []
        
        while uncovered_mask.any() and time.time() < deadline - 15:
            # Randomly pick an uncovered t-subset
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            seed_ts_idx = rng.choice(uncovered_indices.tolist())
            ts = all_ts[seed_ts_idx]
            candidates = ts_to_blocks[ts]
            
            best_idx = -1
            best_gain = -1
            for idx in candidates:
                gain = 0
                for ts_idx in block_to_ts_idx[idx]:
                    if uncovered_mask[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx < 0:
                break
            
            blocks_used.append(best_idx)
            for ts_idx in block_to_ts_idx[best_idx]:
                uncovered_mask[ts_idx] = False
        
        if len(blocks_used) < best_count:
            best_blocks = blocks_used
            best_count = len(blocks_used)
        
        if time.time() < deadline - 10:
            improved = local_search(blocks_used, deadline - 10)
            if len(improved) < best_count:
                best_blocks = improved
                best_count = len(improved)
    
    # Convert block indices back to lists
    if best_blocks is None:
        return []
    
    result = [list(all_blocks[i]) for i in best_blocks]
    # Ensure distinct
    seen = set()
    unique_result = []
    for b in result:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            unique_result.append(b)
    return unique_result
