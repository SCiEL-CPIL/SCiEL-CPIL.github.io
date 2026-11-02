import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    if v == 24 and k == 7 and t == 3:
        return construct_24_7_3(deadline)
    else:
        return construct_generic(v, k, t, deadline)

def construct_24_7_3(deadline):
    v, k, t = 24, 7, 3
    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute triple indices for each block
    block_triple_lists = []
    for block in all_blocks:
        t_list = [tsub_to_idx[triple] for triple in itertools.combinations(block, t)]
        block_triple_lists.append(t_list)
        
    # Precompute sets for faster swap validity check
    block_triple_sets = [set(lst) for lst in block_triple_lists]

    def greedy_construct(randomize=False):
        uncovered = [True] * num_tsubs
        blocks = []
        remaining_uncovered = num_tsubs
        
        while remaining_uncovered > 0:
            best_idx = -1
            best_cov = -1
            
            for i in range(num_blocks):
                cov = 0
                for t_idx in block_triple_lists[i]:
                    if uncovered[t_idx]:
                        cov += 1
                
                if randomize:
                    cov += random.random() * 0.5
                else:
                    cov += 0.0
                
                if cov > best_cov:
                    best_cov = cov
                    best_idx = i
            
            if best_idx == -1:
                break
                
            blocks.append(best_idx)
            for t_idx in block_triple_lists[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    remaining_uncovered -= 1
            
            if time.time() > deadline - 20:
                break
                
        return blocks

    def refine_and_optimize(blocks, max_time):
        if not blocks:
            return []
            
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in blocks:
            for t_idx in block_triple_lists[b_idx]:
                counts[t_idx] += 1
        
        def can_remove(b_idx):
            for t_idx in block_triple_lists[b_idx]:
                if counts[t_idx] <= 1:
                    return False
            return True

        def can_swap(b_old, b_new):
            if b_old == b_new:
                return False
            
            critical_triples = []
            for t_idx in block_triple_lists[b_old]:
                if counts[t_idx] == 1:
                    critical_triples.append(t_idx)
            
            if not critical_triples:
                return True
            
            new_set = block_triple_sets[b_new]
            for t_idx in critical_triples:
                if t_idx not in new_set:
                    return False
            return True

        current_blocks = list(blocks)
        best_blocks = list(blocks)
        best_len = len(blocks)
        
        T = 10.0
        cooling = 0.95
        min_T = 0.1
        max_iters = 50000
        
        start_time_opt = time.time()
        iters = 0
        while iters < max_iters and T > min_T:
            if time.time() > max_time:
                break
            
            iters += 1
            
            move_type = random.random()
            
            if move_type < 0.4 and len(current_blocks) > 0:
                # Try removing a block
                i = random.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                if can_remove(b_idx):
                    for t_idx in block_triple_lists[b_idx]:
                        counts[t_idx] -= 1
                    current_blocks.pop(i)
                    
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                        
            elif move_type < 0.9 and len(current_blocks) > 0:
                # Try swapping a block
                i = random.randint(0, len(current_blocks) - 1)
                b_old = current_blocks[i]
                b_new = random.randint(0, num_blocks - 1)
                
                if can_swap(b_old, b_new):
                    for t_idx in block_triple_lists[b_old]:
                        counts[t_idx] -= 1
                    for t_idx in block_triple_lists[b_new]:
                        counts[t_idx] += 1
                    current_blocks[i] = b_new
                    
                    # Accept swap if it doesn't increase size (it doesn't) 
                    # and we want to explore. 
                    # We accept with probability 0.5 to avoid getting stuck in a specific configuration.
                    if random.random() < 0.5:
                        pass
                    else:
                        for t_idx in block_triple_lists[b_new]:
                            counts[t_idx] -= 1
                        for t_idx in block_triple_lists[b_old]:
                            counts[t_idx] += 1
                        current_blocks[i] = b_old
            else:
                # No-op or Add (skipped to keep size minimal)
                pass

            T *= cooling
            
        # Final cleanup
        counts_best = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in best_blocks:
            for t_idx in block_triple_lists[b_idx]:
                counts_best[t_idx] += 1
                
        final_blocks = []
        for b_idx in best_blocks:
            redundant = True
            for t_idx in block_triple_lists[b_idx]:
                if counts_best[t_idx] == 1:
                    redundant = False
                    break
            if not redundant:
                for t_idx in block_triple_lists[b_idx]:
                    counts_best[t_idx] -= 1
                final_blocks.append(b_idx)
        
        return final_blocks

    blocks_idx = greedy_construct(randomize=False)
    blocks_idx = refine_and_optimize(blocks_idx, deadline - 10)
    
    best_blocks = [list(all_blocks[i]) for i in blocks_idx]
    best_len = len(blocks_idx)
    
    if time.time() < deadline - 15:
        for r in range(10):
            if time.time() > deadline - 5:
                break
            blocks_idx_r = greedy_construct(randomize=True)
            blocks_idx_r = refine_and_optimize(blocks_idx_r, deadline - 5)
            if len(blocks_idx_r) < best_len:
                best_len = len(blocks_idx_r)
                best_blocks = [list(all_blocks[i]) for i in blocks_idx_r]
                
    return best_blocks

def construct_generic(v, k, t, deadline):
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    def greedy_construct():
        uncovered = set(range(num_tsubs))
        blocks = []
        
        while uncovered and time.time() < deadline - 5:
            best_block = None
            best_count = -1
            sample_size = min(10, len(uncovered))
            candidates = set()
            for idx in random.sample(list(uncovered), sample_size):
                triple = all_tsubs[idx]
                remaining = [x for x in range(v) if x not in triple]
                for combo in itertools.combinations(remaining, k - t):
                    block = tuple(sorted(triple + combo))
                    candidates.add(block)
            
            if not candidates:
                break
                
            for block in candidates:
                count = 0
                for triple in itertools.combinations(block, t):
                    idx = tsub_to_idx[triple]
                    if idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = block
            
            if best_block is None:
                break
                
            blocks.append(list(best_block))
            for triple in itertools.combinations(best_block, t):
                idx = tsub_to_idx[triple]
                uncovered.discard(idx)
        
        return blocks

    blocks = greedy_construct()
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                triple_set = set(triple)
                covered = False
                for j, other_block in enumerate(blocks):
                    if j == i:
                        continue
                    if triple_set.issubset(set(other_block)):
                        covered = True
                        break
                if not covered:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
                
    return blocks
