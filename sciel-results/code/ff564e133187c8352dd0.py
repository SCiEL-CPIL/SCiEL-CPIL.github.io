import itertools
import random
import time
import math

def construct(v, k, t):
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    def get_block_triples(block):
        return list(itertools.combinations(block, t))

    def greedy_init():
        uncovered_set = set(range(num_triples))
        blocks = []
        
        while uncovered_set:
            # Pick a random uncovered triple
            start_idx = random.choice(list(uncovered_set))
            start_triple = all_triples[start_idx]
            block = list(start_triple)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                candidates = [p for p in points if p not in block]
                for p in candidates:
                    gain = 0
                    for pair in itertools.combinations(block, t-1):
                        new_triple = tuple(sorted(pair + (p,)))
                        idx = triple_to_idx.get(new_triple)
                        if idx is not None and idx in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                if best_point == -1:
                    best_point = random.choice(candidates)
                block.append(best_point)
            
            block.sort()
            blocks.append(tuple(block))
            
            for tr in get_block_triples(block):
                idx = triple_to_idx[tr]
                if idx in uncovered_set:
                    uncovered_set.remove(idx)
                    
        return blocks

    def sa_optimize(initial_blocks, time_limit=140):
        blocks = list(initial_blocks)
        
        coverage = [0] * num_triples
        for b in blocks:
            for tr in get_block_triples(b):
                coverage[triple_to_idx[tr]] += 1
        
        uncovered_set = set(i for i in range(num_triples) if coverage[i] == 0)
        uncovered_count = len(uncovered_set)
        
        # Cost: Blocks + 100 * Uncovered
        current_cost = len(blocks) + uncovered_count * 100
        best_cost = current_cost
        best_blocks = list(blocks)
        
        start_time = time.time()
        T = 10.0
        cooling = 0.9999
        
        while time.time() - start_time < time_limit:
            move_type = random.random()
            
            if move_type < 0.3 and uncovered_set:
                # Add a block targeting an uncovered triple
                target_idx = random.choice(list(uncovered_set))
                target_triple = all_triples[target_idx]
                existing = set(target_triple)
                candidates = [p for p in points if p not in existing]
                new_points = random.sample(candidates, k - t)
                new_block = tuple(sorted(list(target_triple) + new_points))
                
                # Check if block already exists?
                # If it does, adding it again is redundant but allowed in state?
                # No, blocks must be distinct in final output.
                # In SA state, we can allow duplicates temporarily? 
                # Better to check.
                if new_block in blocks:
                    continue
                    
                blocks.append(new_block)
                
                delta_uncovered = 0
                for tr in get_block_triples(new_block):
                    idx = triple_to_idx[tr]
                    if coverage[idx] == 0:
                        delta_uncovered -= 1
                        uncovered_set.add(idx) # Wait, logic below
                    coverage[idx] += 1
                    if coverage[idx] == 1:
                        uncovered_set.remove(idx)
                        
                # Recalculate delta_uncovered properly?
                # My logic above:
                # If cov was 0, it becomes 1. Uncovered decreases by 1.
                # So delta_uncovered should be - (number of triples that were 0)
                
                # Let's re-calculate delta correctly in the acceptance step
                # Or just update state and re-calculate cost?
                # Re-calculating cost is O(N) = 2024. Too slow for 10M steps.
                # Must use delta.
                
                # Let's trace:
                # Before: cov[idx] = 0. After: cov[idx] = 1.
                # Uncovered count decreases by 1.
                # So delta_uncovered = -1 for each such triple.
                
                # My code above:
                # if coverage[idx] == 0: delta_uncovered -= 1
                # coverage[idx] += 1
                # if coverage[idx] == 1: uncovered_set.remove(idx)
                # This is correct.
                
                delta_cost = 1 + delta_uncovered * 100
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(blocks)
                else:
                    # Revert
                    blocks.pop()
                    for tr in get_block_triples(new_block):
                        idx = triple_to_idx[tr]
                        coverage[idx] -= 1
                        if coverage[idx] == 0:
                            uncovered_set.add(idx)
                            # uncovered_count += 1 (implicit in set size)
                        
            elif move_type < 0.6 and len(blocks) > 1:
                # Remove a block
                idx = random.randint(0, len(blocks) - 1)
                old_block = blocks[idx]
                
                delta_uncovered = 0
                for tr in get_block_triples(old_block):
                    t_idx = triple_to_idx[tr]
                    if coverage[t_idx] == 1:
                        delta_uncovered += 1
                        
                delta_cost = -1 + delta_uncovered * 100
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    blocks.pop(idx)
                    for tr in get_block_triples(old_block):
                        t_idx = triple_to_idx[tr]
                        coverage[t_idx] -= 1
                        if coverage[t_idx] == 0:
                            uncovered_set.add(t_idx)
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(blocks)
                        
            else:
                # Modify a block
                if not blocks:
                    continue
                b_idx = random.randint(0, len(blocks) - 1)
                old_block = blocks[b_idx]
                p_out = random.choice(old_block)
                candidates = [p for p in points if p not in old_block]
                if not candidates:
                    continue
                p_in = random.choice(candidates)
                
                new_block_set = set(old_block)
                new_block_set.remove(p_out)
                new_block_set.add(p_in)
                new_block = tuple(sorted(new_block_set))
                
                if new_block == old_block:
                    continue
                
                old_triples = set(get_block_triples(old_block))
                new_triples = set(get_block_triples(new_block))
                
                removed_triples = old_triples - new_triples
                added_triples = new_triples - old_triples
                
                delta_uncovered = 0
                for tr in removed_triples:
                    idx = triple_to_idx[tr]
                    if coverage[idx] == 1:
                        delta_uncovered += 1
                for tr in added_triples:
                    idx = triple_to_idx[tr]
                    if coverage[idx] == 0:
                        delta_uncovered -= 1
                        
                delta_cost = delta_uncovered * 100
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    blocks[b_idx] = new_block
                    for tr in removed_triples:
                        idx = triple_to_idx[tr]
                        coverage[idx] -= 1
                        if coverage[idx] == 0:
                            uncovered_set.add(idx)
                    for tr in added_triples:
                        idx = triple_to_idx[tr]
                        coverage[idx] += 1
                        if coverage[idx] == 1:
                            uncovered_set.remove(idx)
                            
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(blocks)
                        
            T *= cooling
            if T < 0.01:
                T = 0.01
                
    init_blocks = greedy_init()
    final_blocks = sa_optimize(init_blocks)
    
    # Repair if invalid
    coverage = [0] * num_triples
    for b in final_blocks:
        for tr in get_block_triples(b):
            coverage[triple_to_idx[tr]] += 1
            
    uncovered_set = set(i for i in range(num_triples) if coverage[i] == 0)
    
    if uncovered_set:
        while uncovered_set:
            idx = random.choice(list(uncovered_set))
            tr = all_triples[idx]
            existing = set(tr)
            candidates = [p for p in points if p not in existing]
            block = list(tr)
            while len(block) < k:
                best_p = -1
                best_gain = 0
                for p in candidates:
                    if p in block: continue
                    gain = 0
                    for pair in itertools.combinations(block, t-1):
                        new_tr = tuple(sorted(pair + (p,)))
                        if new_tr in triple_to_idx:
                            n_idx = triple_to_idx[new_tr]
                            if n_idx in uncovered_set:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                if best_p == -1:
                    remaining = [p for p in candidates if p not in block]
                    if remaining:
                        best_p = random.choice(remaining)
                    else:
                        break
                block.append(best_p)
                candidates.remove(best_p)
                
            block.sort()
            new_block = tuple(block)
            final_blocks.append(new_block)
            
            for tr2 in get_block_triples(new_block):
                n_idx = triple_to_idx[tr2]
                coverage[n_idx] += 1
                if coverage[n_idx] == 1:
                    uncovered_set.discard(n_idx)
                    
    return [list(b) for b in final_blocks]
