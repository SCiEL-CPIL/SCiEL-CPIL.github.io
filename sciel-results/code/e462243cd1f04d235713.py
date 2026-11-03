import itertools
import random
import sys

def construct(v, k, t):
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_index = {trip: i for i, trip in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Pre-compute all blocks and their triple lists
    # And reverse mapping: triple -> list of blocks
    all_blocks = []
    block_to_triples = []
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for block in itertools.combinations(points, k):
        block_idx = len(all_blocks)
        all_blocks.append(block)
        trips = []
        for trip in itertools.combinations(block, t):
            idx = triple_to_index[trip]
            trips.append(idx)
            triple_to_blocks[idx].append(block_idx)
        block_to_triples.append(trips)
        
    num_blocks_total = len(all_blocks)
    
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(range(num_triples))
        blocks = []
        
        # Pre-compute block coverage size for speed? No, it changes.
        # But we can use a heuristic.
        
        while uncovered:
            # Pick a random uncovered triple to target
            # This is faster than finding the most constrained one every time
            # and works well with multiple restarts
            target_idx = random.choice(list(uncovered))
            
            # Find best block covering target_idx
            best_block_idx = -1
            max_covered = -1
            
            candidates = triple_to_blocks[target_idx]
            # Randomize candidates to break ties differently
            random.shuffle(candidates)
            
            for b_idx in candidates:
                covered = 0
                for t_idx in block_to_triples[b_idx]:
                    if t_idx in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_block_idx = b_idx
                    if covered == len(block_to_triples[b_idx]):
                        break
            
            if best_block_idx != -1:
                blocks.append(best_block_idx)
                for t_idx in block_to_triples[best_block_idx]:
                    uncovered.discard(t_idx)
                    
        return blocks

    def local_search(blocks):
        # blocks: list of block indices
        # Returns improved list of block indices
        
        # Calculate initial counts
        triple_count = [0] * num_triples
        for b_idx in blocks:
            for t_idx in block_to_triples[b_idx]:
                triple_count[t_idx] += 1
        
        best_blocks = blocks[:]
        best_len = len(blocks)
        
        # Iterative improvement
        # We try to remove blocks and swap blocks to enable removals
        for _ in range(50): # Max iterations to prevent infinite loops
            changed = False
            
            # 1. Remove redundant blocks
            new_blocks = []
            for b_idx in blocks:
                essential = False
                for t_idx in block_to_triples[b_idx]:
                    if triple_count[t_idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove
                    for t_idx in block_to_triples[b_idx]:
                        triple_count[t_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
            
            # 2. Try swaps
            # For each block, see if we can replace it with another that covers its essential triples
            # and potentially allows removing others
            for i in range(len(blocks)):
                b_idx = blocks[i]
                # Find essential triples for b_idx
                essential_triples = []
                for t_idx in block_to_triples[b_idx]:
                    if triple_count[t_idx] == 1:
                        essential_triples.append(t_idx)
                
                if not essential_triples:
                    # Should have been removed, but counts might have changed? 
                    # If no essential triples, it's redundant.
                    for t_idx in block_to_triples[b_idx]:
                        triple_count[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    continue
                
                # Look for a replacement block b_new
                # It must cover all essential_triples
                # We check candidates from the first essential triple
                candidates = triple_to_blocks[essential_triples[0]]
                found = False
                for b_new in candidates:
                    if b_new == b_idx:
                        continue
                    # Check if b_new covers all essential_triples
                    covers_all = True
                    b_new_set = set(block_to_triples[b_new])
                    for t_idx in essential_triples:
                        if t_idx not in b_new_set:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Tentative swap: remove b_idx, add b_new
                    # Check if this leads to a smaller set
                    # 1. Remove b_idx
                    for t_idx in block_to_triples[b_idx]:
                        triple_count[t_idx] -= 1
                    # 2. Add b_new
                    for t_idx in block_to_triples[b_new]:
                        triple_count[t_idx] += 1
                    
                    # 3. Check for new redundancies
                    new_len = len(blocks)
                    # We don't immediately remove others in this check to keep it fast,
                    # but we can check if b_new itself is redundant or if others are.
                    
                    # Simple check: Did we reduce the count?
                    # We replaced 1 block with 1 block. Count is same.
                    # But maybe b_new covers triples from other blocks, making them redundant?
                    
                    # Let's check if any block in blocks (excluding b_idx which is removed, including b_new) is redundant
                    # This is expensive. 
                    
                    # Alternative: Just accept the swap if it doesn't increase size, 
                    # and let the next "Remove redundant" pass clean up.
                    # But we must ensure we don't cycle.
                    
                    # Let's try to remove redundant blocks immediately after swap
                    tentative_blocks = blocks[:]
                    tentative_blocks.remove(b_idx)
                    tentative_blocks.append(b_new)
                    
                    # Check if any block in tentative_blocks is redundant
                    # We can do a quick check
                    removable_found = False
                    for tb_idx in tentative_blocks:
                        is_removable = True
                        for t_idx in block_to_triples[tb_idx]:
                            if triple_count[t_idx] == 1:
                                is_removable = False
                                break
                        if is_removable:
                            removable_found = True
                            break
                    
                    if removable_found:
                        # Accept swap
                        blocks = tentative_blocks
                        changed = True
                        found = True
                        break
                    else:
                        # Rollback
                        for t_idx in block_to_triples[b_new]:
                            triple_count[t_idx] -= 1
                        for t_idx in block_to_triples[b_idx]:
                            triple_count[t_idx] += 1
                        # No change
                        pass
                
                if found:
                    break # Restart loop
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
            
            if not changed:
                break
                
        return best_blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    for seed in range(10):
        initial = greedy_construct(seed)
        improved = local_search(initial)
        
        if len(improved) < best_count:
            best_count = len(improved)
            best_blocks = [list(all_blocks[b]) for b in improved]
            
    return best_blocks
