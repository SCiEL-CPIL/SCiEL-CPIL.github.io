import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (evaluator kills at 150s, but keep buffer)
    
    # Pre-compute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Pre-compute all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    
    # Build coverage matrix: Cover[i, j] = 1 if t-subset j is in block i
    # Using uint8 for memory efficiency
    Cover = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    block_to_tsubs = [None] * num_kblocks
    
    for i, B in enumerate(all_kblocks):
        tsubs = []
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            tsubs.append(j)
        block_to_tsubs[i] = tsubs
    
    # Pre-compute inverse map: for each t-subset, which k-blocks contain it
    # tsub_to_blocks[idx] = list of block indices
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i in range(num_kblocks):
        for idx in block_to_tsubs[i]:
            tsub_to_blocks[idx].append(i)
    
    # Helper: get gain of a block given current uncovered mask
    def get_gain(block_idx, uncovered_mask):
        return int(np.sum(Cover[block_idx] & uncovered_mask))
    
    # Greedy construction using numpy for speed
    uncovered = np.ones(num_tsubs, dtype=np.uint8)
    selected_indices = []
    
    # Greedy phase: pick block covering most uncovered
    while np.sum(uncovered) > 0:
        if time.time() > deadline - 30.0:
            break
        
        # Compute gains for all blocks at once
        gains = Cover @ uncovered
        best_idx = int(np.argmax(gains))
        
        if gains[best_idx] == 0:
            break
        
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
    
    blocks = [all_kblocks[i] for i in selected_indices]
    block_set = set(selected_indices)
    
    # Track best solution
    best_blocks = list(blocks)
    best_len = len(blocks)
    
    # Local optimization phase
    if time.time() < deadline - 10.0:
        # Compute coverage counts
        cover_counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in selected_indices:
            cover_counts += Cover[idx]
        
        # Iteratively remove redundant blocks
        def clean_redundants(block_list, cover_counts, block_set):
            changed = True
            while changed and time.time() < deadline - 10.0:
                changed = False
                new_block_list = []
                new_block_set = set()
                for idx in block_list:
                    # Check if this block covers any t-subset with count 1
                    if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                        new_block_list.append(idx)
                        new_block_set.add(idx)
                    else:
                        # Remove this block
                        cover_counts -= Cover[idx]
                        changed = True
                return new_block_list, new_block_set, cover_counts
        
        selected_indices, block_set, cover_counts = clean_redundants(
            selected_indices, cover_counts, block_set)
        blocks = [all_kblocks[i] for i in selected_indices]
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
        
        # Local search with block replacement
        if time.time() < deadline - 5.0:
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            
            while time.time() < deadline - 5.0:
                if len(selected_indices) == 0:
                    break
                
                # Pick random block to try removing/replacing
                i = random.randint(0, len(selected_indices) - 1)
                b_old_idx = selected_indices[i]
                
                # Find critical t-subsets (those with count 1) covered by this block
                critical_mask = (Cover[b_old_idx] == 1) & (cover_counts == 1)
                critical_tsubs = np.where(critical_mask)[0]
                
                if len(critical_tsubs) == 0:
                    # Block is redundant, remove it
                    selected_indices.pop(i)
                    block_set.discard(b_old_idx)
                    cover_counts -= Cover[b_old_idx]
                    continue
                
                # Try to find an alternative block that covers all critical t-subsets
                best_alt = -1
                best_score = -1
                
                # Sample candidates to speed up
                candidates = tsub_to_blocks[critical_tsubs[0]]
                if len(candidates) > 50:
                    candidates = random.sample(candidates, 50)
                
                for alt_idx in candidates:
                    if alt_idx == b_old_idx or alt_idx in block_set:
                        continue
                    # Check if alt covers all critical t-subsets
                    if np.all(Cover[alt_idx][critical_tsubs] == 1):
                        # Score: prefer blocks that cover more low-count t-subsets
                        score = int(np.sum((Cover[alt_idx] == 1) & (cover_counts <= 2)))
                        if score > best_score:
                            best_score = score
                            best_alt = alt_idx
                
                if best_alt != -1:
                    # Replace
                    cover_counts -= Cover[b_old_idx]
                    cover_counts += Cover[best_alt]
                    selected_indices[i] = best_alt
                    block_set.discard(b_old_idx)
                    block_set.add(best_alt)
                    
                    # Clean up redundants after replacement
                    selected_indices, block_set, cover_counts = clean_redundants(
                        selected_indices, cover_counts, block_set)
                    
                    if len(selected_indices) < best_len:
                        best_len = len(selected_indices)
                        best_blocks = [all_kblocks[idx] for idx in selected_indices]
                else:
                    # No good replacement, try removing and repairing
                    # Remove the block
                    selected_indices.pop(i)
                    block_set.discard(b_old_idx)
                    cover_counts -= Cover[b_old_idx]
                    
                    # Find uncovered t-subsets
                    uncovered_mask = (cover_counts == 0).astype(np.uint8)
                    
                    if np.sum(uncovered_mask) > 0:
                        # Greedy repair: add block covering most uncovered
                        gains = Cover @ uncovered_mask
                        best_idx = int(np.argmax(gains))
                        
                        if gains[best_idx] > 0 and best_idx not in block_set:
                            selected_indices.append(best_idx)
                            block_set.add(best_idx)
                            cover_counts += Cover[best_idx]
                            
                            # Clean up redundants
                            selected_indices, block_set, cover_counts = clean_redundants(
                                selected_indices, cover_counts, block_set)
                            
                            if len(selected_indices) < best_len:
                                best_len = len(selected_indices)
                                best_blocks = [all_kblocks[idx] for idx in selected_indices]
                        else:
                            # Revert
                            selected_indices.append(b_old_idx)
                            block_set.add(b_old_idx)
                            cover_counts += Cover[b_old_idx]
                    
                    # Cool down
                    T *= cooling
                    if T < min_T:
                        T = min_T
    
    return best_blocks


if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    # Verify
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
