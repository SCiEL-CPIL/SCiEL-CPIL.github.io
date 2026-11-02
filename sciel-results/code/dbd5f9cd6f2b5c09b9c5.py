import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    block_subsets = []
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for i, block in enumerate(all_blocks):
        block_sub_ids = []
        for sub in itertools.combinations(block, t):
            sid = subset_to_id[sub]
            block_sub_ids.append(sid)
            subset_to_blocks[sid].append(i)
        block_subsets.append(block_sub_ids)
        
    # 2. Greedy Set Cover with Randomization
    best_solution = None
    best_size = float('inf')
    
    def greedy_run():
        block_gain = [len(block_subsets[i]) for i in range(num_blocks)]
        subset_count = [0] * num_subsets
        selected = [False] * num_blocks
        selected_blocks = []
        
        while True:
            max_gain = -1
            candidates = []
            
            # Find max gain among unselected blocks
            for i in range(num_blocks):
                if selected[i]:
                    continue
                g = block_gain[i]
                if g > max_gain:
                    max_gain = g
                    candidates = [i]
                elif g == max_gain:
                    candidates.append(i)
                    
            if max_gain == 0:
                break
                
            b = random.choice(candidates)
            selected[b] = True
            selected_blocks.append(b)
            block_gain[b] = 0
            
            for sid in block_subsets[b]:
                if subset_count[sid] == 0:
                    for other_b in subset_to_blocks[sid]:
                        if not selected[other_b]:
                            block_gain[other_b] -= 1
                subset_count[sid] += 1
                
        return selected_blocks

    def remove_redundant(blocks):
        if not blocks:
            return []
            
        changed = True
        while changed:
            changed = False
            counts = [0] * num_subsets
            for b in blocks:
                for sid in block_subsets[b]:
                    counts[sid] += 1
            
            new_blocks = []
            for b in blocks:
                is_redundant = True
                for sid in block_subsets[b]:
                    if counts[sid] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for sid in block_subsets[b]:
                        counts[sid] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            
        return blocks

    # Main Loop
    while time.time() < deadline:
        sol = greedy_run()
        sol = remove_redundant(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]
            
    if best_solution is None:
        sol = greedy_run()
        sol = remove_redundant(sol)
        best_solution = sol

    return [list(all_blocks[i]) for i in best_solution]
