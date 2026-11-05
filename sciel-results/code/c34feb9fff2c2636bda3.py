import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute all t-subsets
    points = list(range(v))
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the list of triplet IDs it covers
    block_triplets = []
    for block in all_blocks:
        tids = [triplet_to_id[triplet] for triplet in itertools.combinations(block, t)]
        block_triplets.append(tids)
    
    # Map each triplet to the list of block IDs that cover it
    blocks_covering = [[] for _ in range(n_triplets)]
    for b_idx, tids in enumerate(block_triplets):
        for tid in tids:
            blocks_covering[tid].append(b_idx)
    
    # Precompute block triplets as numpy arrays for speed
    block_triplet_arrays = [np.array(tids, dtype=np.int32) for tids in block_triplets]
    
    def greedy_set_cover(seed=0, ratio_mode=False):
        rng = random.Random(seed)
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        # Initial score: number of triplets in the block
        score = np.array([len(tids) for tids in block_triplets], dtype=np.int32)
        used_blocks = set()
        solution = []
        
        while True:
            if time.time() > deadline - 25:
                break
                
            # Find max score block not in used_blocks
            max_score = -1
            best_block_idx = -1
            candidates = []
            
            for b_idx in range(n_blocks):
                if b_idx not in used_blocks:
                    s = score[b_idx]
                    if s > max_score:
                        max_score = s
                        best_block_idx = b_idx
                        candidates = [b_idx]
                    elif s == max_score:
                        candidates.append(b_idx)
                        
            if best_block_idx == -1:
                break
                
            # Randomly pick from candidates to diversify
            chosen_block = rng.choice(candidates)
            solution.append(chosen_block)
            used_blocks.add(chosen_block)
            
            # Update coverage and scores
            newly_covered = []
            for tid in block_triplets[chosen_block]:
                if covered_count[tid] == 0:
                    newly_covered.append(tid)
                covered_count[tid] += 1
                
            for tid in newly_covered:
                for b_idx in blocks_covering[tid]:
                    score[b_idx] -= 1
                    
            if np.all(covered_count > 0):
                break
                
        return solution

    def local_search_remove(blocks):
        # Re-compute covered_count from scratch
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        for b_idx in blocks:
            for tid in block_triplets[b_idx]:
                covered_count[tid] += 1
                
        new_blocks = []
        changed = True
        while changed:
            changed = False
            next_blocks = []
            for b_idx in blocks:
                # Check if block is redundant
                redundant = True
                for tid in block_triplets[b_idx]:
                    if covered_count[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for tid in block_triplets[b_idx]:
                        covered_count[tid] -= 1
                    changed = True
                else:
                    next_blocks.append(b_idx)
            blocks = next_blocks
        return blocks

    def local_search_swap(blocks, max_swaps=50):
        # Try to replace one block with another that allows removing more blocks
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        for b_idx in blocks:
            for tid in block_triplets[b_idx]:
                covered_count[tid] += 1
        
        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps:
            improved = False
            swap_count += 1
            
            for i, b_idx in enumerate(list(blocks)):
                # Temporarily remove this block
                for tid in block_triplets[b_idx]:
                    covered_count[tid] -= 1
                
                # Find uncovered triplets
                uncovered = np.where(covered_count == 0)[0]
                if len(uncovered) == 0:
                    # Block was redundant, keep it removed
                    blocks = blocks[:i] + blocks[i+1:]
                    improved = True
                    break
                
                # Try to find a single block that covers all uncovered
                best_new_block = -1
                best_gain = -1
                for candidate in range(n_blocks):
                    if candidate in blocks:
                        continue
                    gain = 0
                    for tid in block_triplets[candidate]:
                        if covered_count[tid] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_new_block = candidate
                
                # If adding one block covers all uncovered, do the swap
                if best_gain >= len(uncovered) and best_new_block != -1:
                    blocks = blocks[:i] + [best_new_block] + blocks[i+1:]
                    for tid in block_triplets[best_new_block]:
                        covered_count[tid] += 1
                    improved = True
                    break
                else:
                    # Restore the block
                    for tid in block_triplets[b_idx]:
                        covered_count[tid] += 1
        
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds
    num_seeds = 40
    for seed in range(num_seeds):
        if time.time() > deadline - 35:
            break
        blocks = greedy_set_cover(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
    if best_blocks is None:
        best_blocks = greedy_set_cover(0)
        
    # Local search polish - try on best and a few others
    best_blocks = local_search_remove(best_blocks)
    
    # Try swap local search on the best
    if time.time() < deadline - 15:
        swapped = local_search_swap(best_blocks, max_swaps=30)
        if len(swapped) < len(best_blocks):
            best_blocks = swapped
    
    # Try local search on second-best if time allows
    if time.time() < deadline - 10 and len(best_blocks) > 70:
        # Run one more greedy and try to improve
        alt_blocks = greedy_set_cover(999)
        alt_blocks = local_search_remove(alt_blocks)
        if len(alt_blocks) < len(best_blocks):
            best_blocks = alt_blocks
    
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]

if __name__ == "__main__":
    pass
