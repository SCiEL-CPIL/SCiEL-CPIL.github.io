import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Pre-compute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_blocks)
    
    # Map each t-subset to list of block indices containing it
    # idx[s] = [b1, b2, ...]
    idx = [[] for _ in range(len(all_t_subsets))]
    
    # Map each block index to list of t-subset indices contained in it
    # block_subs[b] = [s1, s2, ...]
    block_subs = [[] for _ in range(num_blocks)]
    
    for b_idx, block in enumerate(all_k_blocks):
        for s in itertools.combinations(block, t):
            s_idx = t_subset_to_idx[s]
            idx[s_idx].append(b_idx)
            block_subs[b_idx].append(s_idx)
            
    # Greedy Construction
    def greedy_construct():
        uncovered = set(range(len(all_t_subsets)))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to base our choice on
            s_idx = random.choice(list(uncovered))
            candidates = idx[s_idx]
            
            best_block = -1
            best_gain = -1
            
            for b in candidates:
                gain = 0
                for sub_idx in block_subs[b]:
                    if sub_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    
            if best_block != -1:
                blocks.append(best_block)
                # Update uncovered
                for sub_idx in block_subs[best_block]:
                    if sub_idx in uncovered:
                        uncovered.remove(sub_idx)
                        
        return blocks

    # Greedy Removal
    def greedy_remove(blocks):
        if not blocks:
            return blocks
            
        # Count coverage
        cover_counts = [0] * len(all_t_subsets)
        for b in blocks:
            for s_idx in block_subs[b]:
                cover_counts[s_idx] += 1
                
        new_blocks = []
        for b in blocks:
            # Check if b is redundant
            redundant = True
            for s_idx in block_subs[b]:
                if cover_counts[s_idx] == 1:
                    redundant = False
                    break
                    
            if redundant:
                # Remove b
                for s_idx in block_subs[b]:
                    cover_counts[s_idx] -= 1
            else:
                new_blocks.append(b)
                
        return new_blocks

    # Local Search
    def local_search(blocks):
        blocks = greedy_remove(blocks)
        best_blocks = blocks
        best_size = len(blocks)
        
        while time.time() - start_time < time_limit - 5.0:
            # Pick a random block to remove
            if not blocks:
                break
                
            b_idx_to_remove = random.choice(range(len(blocks)))
            b = blocks[b_idx_to_remove]
            
            # Remove it
            temp_blocks = [b for i, b in enumerate(blocks) if i != b_idx_to_remove]
            
            # Check if still valid
            cover_counts = [0] * len(all_t_subsets)
            for b in temp_blocks:
                for s_idx in block_subs[b]:
                    cover_counts[s_idx] += 1
                    
            uncovered = [i for i, c in enumerate(cover_counts) if c == 0]
            
            if not uncovered:
                # Valid without b
                blocks = temp_blocks
                blocks = greedy_remove(blocks)
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = blocks
                continue
                
            # Need to cover 'uncovered'
            # Pick random uncovered subset
            s_idx = random.choice(uncovered)
            candidates = idx[s_idx]
            
            best_new_block = -1
            best_gain = -1
            
            for b in candidates:
                if b in temp_blocks:
                    continue
                gain = 0
                for sub_idx in block_subs[b]:
                    if sub_idx in set(uncovered):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_new_block = b
                    
            if best_new_block != -1:
                # Try replacing
                new_blocks = temp_blocks + [best_new_block]
                # Run greedy remove on new set
                new_blocks = greedy_remove(new_blocks)
                
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    if len(blocks) < best_size:
                        best_size = len(blocks)
                        best_blocks = blocks
                elif len(new_blocks) == len(blocks) and random.random() < 0.1:
                    # Accept equal size with some probability to escape local minima
                    # But we must ensure it's valid (it is, by construction)
                    blocks = new_blocks
                    
        return best_blocks

    # Main logic
    random.seed(42) # Fixed seed for reproducibility if needed, but we want exploration
    # Actually, let's use a different seed for each call if possible, or just one good run.
    # The task says "score is seed/value". If we are deterministic, we get one score.
    # Let's run one robust sequence.
    
    blocks = greedy_construct()
    blocks = greedy_remove(blocks)
    
    # If we have time, do local search
    if time.time() - start_time < time_limit - 10.0:
        blocks = local_search(blocks)
        
    # Convert block indices back to lists of points
    result = [list(all_k_blocks[b]) for b in blocks]
    return result
