import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Solve the covering design problem C(v, k, t) using a randomized greedy 
    set cover approach followed by local search for block removal and swapping.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin from 120s limit
    
    # Precompute all t-subsets (the universe to be covered)
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each potential block (k-subset), which t-subsets it covers
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_covered_t = [None] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, b in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_covered_t[i] = covered_indices
    
    # Greedy Cover Function
    def run_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered_set = set(range(num_t_subsets))
        selected = []
        
        while uncovered_set:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset to focus on
            target_t_idx = random.choice(list(uncovered_set))
            candidates = t_to_blocks[target_t_idx]
            
            best_block = -1
            best_gain = -1
            best_blocks_with_gain = []
            
            for b_idx in candidates:
                gain = 0
                for t_idx in block_covered_t[b_idx]:
                    if t_idx in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_blocks_with_gain = [b_idx]
                elif gain == best_gain:
                    best_blocks_with_gain.append(b_idx)
            
            if best_block == -1:
                break
            
            # Randomly select from best blocks to add diversity
            best_block = random.choice(best_blocks_with_gain)
            selected.append(best_block)
            
            # Update uncovered set
            for t_idx in block_covered_t[best_block]:
                if t_idx in uncovered_set:
                    uncovered_set.discard(t_idx)
        
        return selected
    
    # Local Search: Remove Redundant Blocks
    def try_remove_blocks(blocks):
        if not blocks:
            return blocks
        
        # Build coverage counts
        cover_count = [0] * num_t_subsets
        for b_idx in blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        # Greedy removal:
        # Iterate through blocks. If a block is redundant, remove it.
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 10.0:
                    return blocks
                
                b_idx = blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block i
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break  # Restart loop since indices changed
        
        return blocks
    
    # Local Search: Swap-based Improvement
    def local_search_swap(blocks, max_iters=1000):
        """
        Try to improve the solution by swapping one block with another
        that covers all previously uncovered t-subsets while possibly
        allowing removal of other redundant blocks.
        """
        if not blocks:
            return blocks
        
        current_blocks = list(blocks)
        
        # Build coverage counts
        cover_count = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iters and time.time() < deadline - 15.0:
            improved = False
            iterations += 1
            
            # First, try to remove redundant blocks
            removed_any = False
            for i in range(len(current_blocks)):
                if time.time() > deadline - 15.0:
                    break
                b_idx = current_blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    removed_any = True
                    break
            
            if removed_any:
                continue
            
            # Try swapping: pick a block, try to replace it with a different
            # block that covers all its essential t-subsets (those only it covers)
            for i in range(len(current_blocks)):
                if time.time() > deadline - 15.0:
                    break
                
                b_idx = current_blocks[i]
                
                # Find t-subsets only covered by this block
                essential_t = []
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] == 1:
                        essential_t.append(t_idx)
                
                # If no essential t-subsets, this block is redundant (shouldn't happen after removal pass)
                if not essential_t:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement block that covers all essential_t
                # and has a "gain" (covers more other t-subsets)
                best_swap_block = -1
                best_swap_gain = -1
                
                # Sample some candidate blocks for efficiency
                candidates = random.sample(range(num_blocks), min(5000, num_blocks))
                
                for cand_idx in candidates:
                    if cand_idx == b_idx:
                        continue
                    
                    # Check if candidate covers all essential t-subsets
                    covers_all = True
                    gain = 0
                    for t_idx in block_covered_t[cand_idx]:
                        if t_idx in essential_t:
                            # This is essential, must be covered
                            pass
                        if cover_count[t_idx] == 0:
                            gain += 1
                    
                    # More efficient check: does candidate cover all essential_t?
                    for t_idx in essential_t:
                        if t_idx not in block_covered_t[cand_idx]:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Calculate gain: how many currently uncovered t-subsets does it cover?
                    # (After removing b_idx, essential_t would still be covered by cand_idx)
                    # Additional benefit: covers other t-subsets that have low coverage count
                    swap_gain = 0
                    for t_idx in block_covered_t[cand_idx]:
                        if t_idx not in block_covered_t[b_idx]:
                            if cover_count[t_idx] <= 2:
                                swap_gain += 1
                    
                    if swap_gain > best_swap_gain:
                        best_swap_gain = swap_gain
                        best_swap_block = cand_idx
                
                if best_swap_block != -1 and best_swap_gain > 0:
                    # Perform the swap
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_covered_t[best_swap_block]:
                        cover_count[t_idx] += 1
                    current_blocks[i] = best_swap_block
                    improved = True
                    break
        
        return current_blocks
    
    # Main Search Loop
    best_blocks = None
    best_len = float('inf')
    
    # Try a few deterministic seeds first
    seeds = [42, 123, 999, 0, 1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23]
    
    for seed in seeds:
        if time.time() > deadline - 25.0:
            break
        
        greedy_res = run_greedy(seed)
        if not greedy_res:
            continue
        
        # Post-process: first remove redundant blocks, then try local search
        processed_res = try_remove_blocks(greedy_res)
        ls_res = local_search_swap(processed_res)
        
        if len(ls_res) < best_len:
            best_len = len(ls_res)
            best_blocks = ls_res
        
        if best_len <= 69:
            break
    
    # If we have time, try more random seeds
    if time.time() < deadline - 20.0:
        while time.time() < deadline - 10.0:
            seed = random.randint(0, 1000000)
            greedy_res = run_greedy(seed)
            if not greedy_res:
                continue
            
            processed_res = try_remove_blocks(greedy_res)
            ls_res = local_search_swap(processed_res)
            
            if len(ls_res) < best_len:
                best_len = len(ls_res)
                best_blocks = ls_res
                
            if best_len <= 69:
                break
    
    if best_blocks is None:
        return []
    
    # Convert block indices back to the actual blocks (lists of points)
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    # Ensure distinctness and sorting
    for b in final_blocks:
        b.sort()
    
    return final_blocks
