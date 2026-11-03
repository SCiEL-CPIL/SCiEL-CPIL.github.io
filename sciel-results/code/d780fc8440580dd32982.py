"""Optimized covering C(24,7,3) using Weighted Greedy and Counting-Based Local Search."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using:
    1. MRV-Greedy: Select blocks based on the most constrained uncovered triple.
    2. Counting-Based Local Search: Efficiently remove redundant blocks and perform swaps
       using cover counts to maintain state in O(1) per triple update.
    """
    # Pre-compute triple IDs and block-triple mappings for speed
    triple_to_id = {}
    id_to_triple = []
    for triple in itertools.combinations(range(v), t):
        triple_to_id[triple] = len(id_to_triple)
        id_to_triple.append(triple)
    
    total_triples = len(id_to_triple)
    
    # Phase 1: MRV-Greedy
    blocks = _mrv_greedy(v, k, t, triple_to_id, id_to_triple)
    
    # Phase 2: Counting-Based Local Search
    blocks = _counting_local_search(v, k, t, blocks, triple_to_id, id_to_triple)
    
    return blocks

def _mrv_greedy(v, k, t, triple_to_id, id_to_triple):
    """
    Greedy algorithm that picks a block containing the 'most isolated' uncovered triple.
    """
    uncovered = set(range(len(id_to_triple)))
    blocks = []
    
    # Helper to get blocks containing a specific triple
    def get_blocks_containing(triple):
        remaining_points = [p for p in range(v) if p not in triple]
        for combo in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(triple + combo))
            yield block

    while uncovered:
        # Calculate point frequency in uncovered set
        point_freq = [0] * v
        for tid in uncovered:
            for p in id_to_triple[tid]:
                point_freq[p] += 1
        
        best_tid = None
        min_freq_sum = float('inf')
        for tid in uncovered:
            freq_sum = sum(point_freq[p] for p in id_to_triple[tid])
            if freq_sum < min_freq_sum:
                min_freq_sum = freq_sum
                best_tid = tid
        
        critical_triple = id_to_triple[best_tid]
        
        # Evaluate all blocks containing critical_triple
        best_block = None
        best_gain = -1
        
        for block in get_blocks_containing(critical_triple):
            gain = 0
            for triple in itertools.combinations(block, t):
                tid = triple_to_id[triple]
                if tid in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            break
        
        blocks.append(list(best_block))
        
        for triple in itertools.combinations(best_block, t):
            tid = triple_to_id[triple]
            uncovered.discard(tid)
            
    return blocks

def _counting_local_search(v, k, t, blocks, triple_to_id, id_to_triple, max_iters=100):
    """
    Local search using cover counts.
    """
    blocks = [tuple(b) for b in blocks]
    total_triples = len(id_to_triple)
    
    def get_block_triples(block):
        return [triple_to_id[triple] for triple in itertools.combinations(block, t)]
    
    cover_count = [0] * total_triples
    block_triples_list = []
    for b in blocks:
        t_ids = get_block_triples(b)
        block_triples_list.append(t_ids)
        for tid in t_ids:
            cover_count[tid] += 1
            
    def remove_block(i):
        for tid in block_triples_list[i]:
            cover_count[tid] -= 1
        blocks.pop(i)
        block_triples_list.pop(i)
        
    # Pass 1: Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            redundant = True
            for tid in block_triples_list[i]:
                if cover_count[tid] == 1:
                    redundant = False
                    break
            if redundant:
                remove_block(i)
                changed = True
                break

    # Pass 2: Swap and Reduce
    random.seed(42)
    for _ in range(max_iters):
        if not blocks:
            break
            
        i = random.randint(0, len(blocks) - 1)
        old_block = blocks[i]
        
        redundant = True
        for tid in block_triples_list[i]:
            if cover_count[tid] == 1:
                redundant = False
                break
        
        if redundant:
            remove_block(i)
            continue
        
        U_i = []
        for tid in block_triples_list[i]:
            if cover_count[tid] == 1:
                U_i.append(tid)
                
        if not U_i:
            continue
            
        points_in_Ui = set()
        for tid in U_i:
            points_in_Ui.update(id_to_triple[tid])
            
        if len(points_in_Ui) > k:
            continue
            
        success = False
        for attempt in range(10):
            B_new_set = set(points_in_Ui)
            while len(B_new_set) < k:
                candidates = [p for p in range(v) if p not in B_new_set]
                if not candidates: break
                B_new_set.add(random.choice(candidates))
                
            if len(B_new_set) != k:
                continue
                
            B_new = tuple(sorted(B_new_set))
            
            if B_new == old_block:
                continue
                
            covers_Ui = True
            B_new_set_check = set(B_new)
            for tid in U_i:
                if not set(id_to_triple[tid]).issubset(B_new_set_check):
                    covers_Ui = False
                    break
                    
            if not covers_Ui:
                continue
                
            # Perform Swap
            for tid in block_triples_list[i]:
                cover_count[tid] -= 1
            blocks[i] = B_new
            new_t_ids = get_block_triples(B_new)
            block_triples_list[i] = new_t_ids
            for tid in new_t_ids:
                cover_count[tid] += 1
                
            old_len = len(blocks)
            changed = True
            while changed:
                changed = False
                for j in range(len(blocks)):
                    redundant = True
                    for tid in block_triples_list[j]:
                        if cover_count[tid] == 1:
                            redundant = False
                            break
                    if redundant:
                        remove_block(j)
                        changed = True
                        break
                        
            if len(blocks) < old_len:
                success = True
                break
            else:
                for tid in new_t_ids:
                    cover_count[tid] -= 1
                blocks[i] = old_block
                block_triples_list[i] = get_block_triples(old_block)
                for tid in block_triples_list[i]:
                    cover_count[tid] += 1
                    
    return [list(b) for b in blocks]
