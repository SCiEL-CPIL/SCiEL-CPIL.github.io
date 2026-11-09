import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Precomputation
    all_targets = list(itertools.combinations(range(v), t))
    target_index = {tup: i for i, tup in enumerate(all_targets)}
    num_targets = len(all_targets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage for each block as a set of target indices
    block_coverage = []
    for b in all_blocks:
        covered = set()
        for sub in itertools.combinations(b, t):
            covered.add(target_index[sub])
        block_coverage.append(covered)

    # Greedy Construction
    uncovered = set(range(num_targets))
    blocks_idx = []
    used = set()
    
    while uncovered:
        if time.time() > deadline - 30:  # Leave 30s for removal/search
            break
        
        best_idx = -1
        best_gain = -1
        for i, cov_set in enumerate(block_coverage):
            if i in used:
                continue
            gain = len(cov_set & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == len(cov_set):
                    break
                    
        if best_idx == -1:
            break # Should not happen
            
        blocks_idx.append(best_idx)
        used.add(best_idx)
        uncovered -= block_coverage[best_idx]
        
    # Removal Phase
    cover_count = [0] * num_targets
    for idx in blocks_idx:
        for t in block_coverage[idx]:
            cover_count[t] += 1
            
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        to_remove = []
        for i in range(len(blocks_idx)):
            idx = blocks_idx[i]
            redundant = True
            for t in block_coverage[idx]:
                if cover_count[t] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        for i in reversed(to_remove):
            idx = blocks_idx[i]
            for t in block_coverage[idx]:
                cover_count[t] -= 1
            blocks_idx.pop(i)
            changed = True
    
    # Local Search Phase - try to swap blocks to enable more removals
    best_len = len(blocks_idx)
    best_blocks = blocks_idx[:]
    
    # Build a map from target to list of blocks covering it
    target_to_blocks = [[] for _ in range(num_targets)]
    for i, idx in enumerate(blocks_idx):
        for t in block_coverage[idx]:
            target_to_blocks[t].append(i)
    
    while time.time() < deadline - 5:
        # Try to find a pair of blocks where swapping one for another enables a removal
        if not blocks_idx:
            break
            
        # Pick a random block to potentially replace
        i = random.randint(0, len(blocks_idx) - 1)
        idx = blocks_idx[i]
        
        # Find targets that would become uncovered if we remove this block
        would_become_uncovered = []
        for t in block_coverage[idx]:
            if len(target_to_blocks[t]) == 1:
                would_become_uncovered.append(t)
                
        if not would_become_uncovered:
            # This block is redundant, should have been removed
            continue
            
        # Find a replacement block that covers all would_become_uncovered targets
        # and also covers some other targets to potentially enable further removals
        best_replacement = -1
        best_score = -1
        
        for j, cov_set in enumerate(block_coverage):
            if j in used:
                continue
            # Check if this block covers all would_become_uncovered targets
            if all(t in cov_set for t in would_become_uncovered):
                # Score: how many additional targets does this cover that are currently
                # covered by only one block (enabling potential removals)
                score = 0
                for t in cov_set:
                    if t not in block_coverage[idx] and len(target_to_blocks[t]) == 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_replacement = j
        
        if best_replacement != -1 and best_score > 0:
            # Try the replacement
            old_idx = blocks_idx[i]
            blocks_idx[i] = best_replacement
            used.remove(old_idx)
            used.add(best_replacement)
            
            # Update cover counts and target_to_blocks
            for t in block_coverage[old_idx]:
                cover_count[t] -= 1
                target_to_blocks[t].remove(i)
            for t in block_coverage[best_replacement]:
                cover_count[t] += 1
                target_to_blocks[t].append(i)
            
            # Try to remove blocks again
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                to_remove = []
                for idx2 in range(len(blocks_idx)):
                    idx = blocks_idx[idx2]
                    redundant = True
                    for t in block_coverage[idx]:
                        if cover_count[t] == 1:
                            redundant = False
                            break
                    if redundant:
                        to_remove.append(idx2)
                
                for idx2 in reversed(to_remove):
                    idx = blocks_idx[idx2]
                    for t in block_coverage[idx]:
                        cover_count[t] -= 1
                        target_to_blocks[t].remove(idx2)
                    blocks_idx.pop(idx2)
                    changed = True
            
            if len(blocks_idx) < best_len:
                best_len = len(blocks_idx)
                best_blocks = blocks_idx[:]
    
    # Return the best solution found
    return [list(all_blocks[i]) for i in best_blocks]
