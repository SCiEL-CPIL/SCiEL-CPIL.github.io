import itertools
import time
import sys
import heapq
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    points = range(v)
    targets = list(itertools.combinations(points, t))
    blocks = list(itertools.combinations(points, k))
    
    num_targets = len(targets)
    num_blocks = len(blocks)
    
    target_index = {t: i for i, t in enumerate(targets)}
    
    # Precompute relationships
    block_to_targets = [[] for _ in range(num_blocks)]
    target_to_blocks = [[] for _ in range(num_targets)]
    
    for j, block in enumerate(blocks):
        for target in itertools.combinations(block, t):
            i = target_index[target]
            block_to_targets[j].append(i)
            target_to_blocks[i].append(j)
            
    # Helper to check time
    def time_left():
        return deadline - time.time()
        
    # Greedy Construction
    def greedy_construct():
        uncovered = [True] * num_targets
        counts = [len(block_to_targets[j]) for j in range(num_blocks)]
        
        # Heap items: (-count, random_tiebreaker, block_idx)
        heap = [(-counts[j], random.random(), j) for j in range(num_blocks)]
        heapq.heapify(heap)
        
        selected = set()
        covered_count = 0
        
        while covered_count < num_targets:
            if time_left() < 10: # Safety margin
                break
                
            best_count = -1
            best_idx = -1
            
            while heap:
                neg_c, _, idx = heapq.heappop(heap)
                c = -neg_c
                # Check if this entry is stale
                if c == counts[idx] and idx not in selected:
                    best_count = c
                    best_idx = idx
                    break
                # If stale, discard
                
            if best_idx == -1:
                # Heap exhausted or no valid blocks found?
                break
                
            selected.add(best_idx)
            
            # Update counts for targets covered by best_idx
            for i in block_to_targets[best_idx]:
                if uncovered[i]:
                    uncovered[i] = False
                    covered_count += 1
                    # Update all blocks covering this target
                    for k_idx in target_to_blocks[i]:
                        counts[k_idx] -= 1
                        heapq.heappush(heap, (-counts[k_idx], random.random(), k_idx))
                        
        return list(selected)

    def local_search(selected_blocks):
        selected = set(selected_blocks)
        
        # Calculate initial counts
        counts = [0] * num_targets
        for b in selected:
            for t_idx in block_to_targets[b]:
                counts[t_idx] += 1
                
        improved = True
        while improved and time_left() > 5:
            improved = False
            
            # Phase 1: Reduction
            changed = True
            while changed and time_left() > 5:
                changed = False
                # Iterate over a copy to allow removal
                for b in list(selected):
                    # Check if b is redundant
                    redundant = True
                    for t_idx in block_to_targets[b]:
                        if counts[t_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        selected.remove(b)
                        for t_idx in block_to_targets[b]:
                            counts[t_idx] -= 1
                        changed = True
                        improved = True
            
            # Phase 2: Swapping
            if time_left() < 10:
                break
                
            # Try to swap each block
            blocks_to_try = list(selected)
            random.shuffle(blocks_to_try)
            
            for b_old in blocks_to_try:
                if time_left() < 10:
                    break
                if b_old not in selected:
                    continue
                    
                # Temporarily remove b_old
                for t_idx in block_to_targets[b_old]:
                    counts[t_idx] -= 1
                    
                # Find critical targets
                critical_targets = []
                for t_idx in block_to_targets[b_old]:
                    if counts[t_idx] == 0:
                        critical_targets.append(t_idx)
                        
                if not critical_targets:
                    # Any block is a valid replacement
                    # We want one that maximizes "slack" (covers many count-1 targets)
                    best_score = -1
                    best_new = -1
                    # Sample some random blocks
                    candidates = random.sample(range(num_blocks), min(1000, num_blocks))
                    for b_new in candidates:
                        if b_new in selected:
                            continue
                        score = 0
                        for t_idx in block_to_targets[b_new]:
                            if counts[t_idx] == 1:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_new = b_new
                            
                    if best_new != -1:
                        selected.add(best_new)
                        for t_idx in block_to_targets[best_new]:
                            counts[t_idx] += 1
                        improved = True
                    else:
                        # No good swap found, revert
                        for t_idx in block_to_targets[b_old]:
                            counts[t_idx] += 1
                else:
                    # Must cover critical_targets
                    # Find blocks containing all critical_targets
                    if len(critical_targets) == 1:
                        ct = critical_targets[0]
                        candidates = target_to_blocks[ct]
                    else:
                        # Sort by list length to minimize intersection size
                        ct_sorted = sorted(critical_targets, key=lambda x: len(target_to_blocks[x]))
                        candidates = set(target_to_blocks[ct_sorted[0]])
                        for ct in ct_sorted[1:]:
                            candidates &= set(target_to_blocks[ct])
                            if not candidates:
                                break
                            
                    if candidates:
                        # Filter out selected blocks
                        candidates -= selected
                        
                        # Pick best candidate
                        best_score = -1
                        best_new = -1
                        # If too many candidates, sample
                        if len(candidates) > 500:
                            candidates = set(random.sample(list(candidates), 500))
                        
                        for b_new in candidates:
                            score = 0
                            for t_idx in block_to_targets[b_new]:
                                if counts[t_idx] == 1:
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_new = b_new
                                
                        if best_new != -1:
                            selected.add(best_new)
                            for t_idx in block_to_targets[best_new]:
                                counts[t_idx] += 1
                            improved = True
                        else:
                            # Revert
                            for t_idx in block_to_targets[b_old]:
                                counts[t_idx] += 1
                    else:
                        # No candidate found, revert
                        for t_idx in block_to_targets[b_old]:
                            counts[t_idx] += 1
                        
        return list(selected)

    # Main Loop
    best_result = None
    best_len = float('inf')
    
    # Run multiple restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time_left() < 20:
            break
            
        random.seed(42 + r)
        
        # Greedy
        initial = greedy_construct()
        
        # Local Search
        refined = local_search(initial)
        
        if len(refined) < best_len:
            best_len = len(refined)
            best_result = refined
            
    if best_result is None:
        # Fallback: return the first greedy result if no time for LS
        initial = greedy_construct()
        return [list(blocks[b]) for b in initial]
        
    return [list(blocks[b]) for b in best_result]
