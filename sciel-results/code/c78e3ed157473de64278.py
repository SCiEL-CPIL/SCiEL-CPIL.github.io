import itertools
import random
import time
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin out of 150s

    # Pre-computation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Map each block to the list of triple indices it covers
    block_triples = []
    for b in all_blocks:
        tris = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]
        block_triples.append(tris)

    # Map each triple to the list of block indices that cover it
    triple_to_blocks = defaultdict(list)
    for b_idx, tris in enumerate(block_triples):
        for tri_idx in tris:
            triple_to_blocks[tri_idx].append(b_idx)

    def greedy_construct():
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered and time.time() < deadline - 1:
            # Pick the uncovered triple with the fewest covering blocks
            best_tri = None
            best_count = float('inf')
            sample = list(uncovered)
            random.shuffle(sample)
            for tri_idx in sample[:50]:  # Sample up to 50 for speed
                cnt = len(triple_to_blocks[tri_idx])
                if cnt < best_count:
                    best_count = cnt
                    best_tri = tri_idx
            
            if best_tri is None:
                best_tri = random.choice(list(uncovered))
            
            candidates = triple_to_blocks[best_tri]
            best_block = None
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for tri in block_triples[b_idx]:
                    if tri in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block is not None:
                blocks.append(best_block)
                for tri in block_triples[best_block]:
                    if tri in uncovered:
                        uncovered.remove(tri)
        
        return blocks

    def optimize_blocks(blocks):
        if not blocks:
            return []
            
        # counts[tri_idx] = number of blocks covering tri_idx
        counts = [0] * n_triples
        for b_idx in blocks:
            for tri in block_triples[b_idx]:
                counts[tri] += 1
                
        active = [True] * len(blocks)
        
        def is_redundant(i):
            b_idx = blocks[i]
            for tri in block_triples[b_idx]:
                if counts[tri] <= 1:
                    return False
            return True

        def remove_block(i):
            active[i] = False
            b_idx = blocks[i]
            for tri in block_triples[b_idx]:
                counts[tri] -= 1

        # Removal phase
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if not active[i]:
                    continue
                if is_redundant(i):
                    remove_block(i)
                    changed = True
        
        # Swap phase
        no_improvement_count = 0
        
        while time.time() < deadline - 1:
            # Try removal again
            pass_removal = True
            while pass_removal and time.time() < deadline - 1:
                pass_removal = False
                indices = list(range(len(blocks)))
                random.shuffle(indices)
                for i in indices:
                    if not active[i]:
                        continue
                    if is_redundant(i):
                        remove_block(i)
                        pass_removal = True
                        changed = True
                        
            active_indices = [i for i in range(len(blocks)) if active[i]]
            if not active_indices:
                break
                
            i_out = random.choice(active_indices)
            b_out = blocks[i_out]
            
            active[i_out] = False
            for tri in block_triples[b_out]:
                counts[tri] -= 1
                
            uncovered_tris = []
            for tri_idx in range(n_triples):
                if counts[tri_idx] == 0:
                    uncovered_tris.append(tri_idx)
                        
            if len(uncovered_tris) > 10:
                active[i_out] = True
                for tri in block_triples[b_out]:
                    counts[tri] += 1
                no_improvement_count += 1
                if no_improvement_count > 100:
                    break
                continue
                
            if not uncovered_tris:
                pass
            else:
                candidates = set(triple_to_blocks[uncovered_tris[0]])
                for tri in uncovered_tris[1:]:
                    candidates.intersection_update(triple_to_blocks[tri])
                    if not candidates:
                        break
                    
                if candidates:
                    current_block_set = set(blocks[i] for i in range(len(blocks)) if active[i])
                    valid_candidates = [b for b in candidates if b != b_out and b not in current_block_set]
                    
                    if valid_candidates:
                        b_in = random.choice(valid_candidates)
                        blocks.append(b_in)
                        i_in = len(blocks) - 1
                        active.append(True)
                        for tri in block_triples[b_in]:
                            counts[tri] += 1
                        no_improvement_count = 0
                        changed = True
                        continue
                    else:
                        active[i_out] = True
                        for tri in block_triples[b_out]:
                            counts[tri] += 1
                        no_improvement_count += 1
                else:
                    active[i_out] = True
                    for tri in block_triples[b_out]:
                        counts[tri] += 1
                    no_improvement_count += 1
                    
                if no_improvement_count > 100:
                    break

        final_blocks = [blocks[i] for i in range(len(blocks)) if active[i]]
        return final_blocks

    def simulated_annealing(init_blocks, time_limit):
        if not init_blocks:
            return init_blocks
        
        current_blocks = list(init_blocks)
        current_counts = [0] * n_triples
        for b_idx in current_blocks:
            for tri in block_triples[b_idx]:
                current_counts[tri] += 1
        
        current_cost = len(current_blocks)
        best_cost = current_cost
        best_blocks = list(current_blocks)
        
        current_block_set = set(current_blocks)
        
        T = 5.0
        alpha = 0.95
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Pick a block to remove
            idx_out = random.randint(0, len(current_blocks) - 1)
            b_out_idx = current_blocks[idx_out]
            
            # Temporarily remove
            for tri in block_triples[b_out_idx]:
                current_counts[tri] -= 1
            
            # Find uncovered
            uncovered = []
            for tri_idx in range(n_triples):
                if current_counts[tri_idx] == 0:
                    uncovered.append(tri_idx)
            
            # Find a block that covers all uncovered
            if not uncovered:
                # It was redundant
                current_blocks.pop(idx_out)
                current_block_set.discard(b_out_idx)
                current_cost -= 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = list(current_blocks)
                T = max(T * alpha, 0.1)
                continue
            
            # Try to find a single block that covers all uncovered
            if len(uncovered) <= 5:
                candidates = set(triple_to_blocks[uncovered[0]])
                for tri in uncovered[1:]:
                    candidates.intersection_update(triple_to_blocks[tri])
                    if not candidates:
                        break
                
                valid_cands = [b for b in candidates if b != b_out_idx and b not in current_block_set]
                
                if valid_cands:
                    b_in_idx = random.choice(valid_cands)
                    current_blocks[idx_out] = b_in_idx
                    current_block_set.discard(b_out_idx)
                    current_block_set.add(b_in_idx)
                    
                    for tri in block_triples[b_in_idx]:
                        current_counts[tri] += 1
                    
                    new_uncovered = sum(1 for c in current_counts if c == 0)
                    new_cost = len(current_blocks) + 100 * new_uncovered
                    
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < min(1.0, max(0.0, -delta / T)):
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                    else:
                        # Revert
                        for tri in block_triples[b_out_idx]:
                            current_counts[tri] += 1
                        current_blocks[idx_out] = b_out_idx
                        current_block_set.discard(b_in_idx)
                        current_block_set.add(b_out_idx)
                else:
                    # Revert
                    for tri in block_triples[b_out_idx]:
                        current_counts[tri] += 1
            else:
                # Revert
                for tri in block_triples[b_out_idx]:
                    current_counts[tri] += 1
            
            T = max(T * alpha, 0.1)
        
        return best_blocks

    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Greedy init
        cand = greedy_construct()
        
        # Optimize
        opt_cand = optimize_blocks(cand)
        
        if len(opt_cand) < best_len:
            best_len = len(opt_cand)
            best_blocks = opt_cand
        
        if best_len <= 69:
            break
        
        # If time remains, try SA
        remaining = deadline - time.time()
        if remaining > 10:
            sa_result = simulated_annealing(opt_cand, remaining - 2)
            if len(sa_result) < best_len:
                best_len = len(sa_result)
                best_blocks = sa_result
    
    if best_blocks is None:
        best_blocks = greedy_construct()
    
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]
