import itertools
import random
import time
import math

def construct(v, k, t):
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N = len(all_blocks)
    
    all_targets = list(itertools.combinations(points, t))
    M = len(all_targets)
    
    target_index = {target: i for i, target in enumerate(all_targets)}
    
    # Precompute subsets for each block
    block_subsets = []
    for block in all_blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(target_index[sub])
        block_subsets.append(subs)
        
    # Precompute target_to_blocks for initial greedy
    target_to_blocks = [[] for _ in range(M)]
    for i, subs in enumerate(block_subsets):
        for j in subs:
            target_to_blocks[j].append(i)
            
    random.seed(42)
    
    # State variables
    cov = [0] * M
    x = [0] * N
    selected_indices = []  # List of currently selected block indices
    selected_count = 0
    uncovered_count = M
    
    def add_block(i):
        nonlocal selected_count, uncovered_count
        if x[i]:
            return
        x[i] = 1
        selected_indices.append(i)
        selected_count += 1
        for s in block_subsets[i]:
            if cov[s] == 0:
                uncovered_count -= 1
            cov[s] += 1
            
    def remove_block(i):
        nonlocal selected_count, uncovered_count
        if not x[i]:
            return
        x[i] = 0
        # Remove from selected_indices efficiently
        try:
            idx = selected_indices.index(i)
            selected_indices[idx] = selected_indices[-1]
            selected_indices.pop()
        except ValueError:
            pass
        selected_count -= 1
        for s in block_subsets[i]:
            cov[s] -= 1
            if cov[s] == 0:
                uncovered_count += 1
                
    # 1. Initialization: Greedy
    while uncovered_count > 0:
        # Find an uncovered target
        t_idx = -1
        for i in range(M):
            if cov[i] == 0:
                t_idx = i
                break
        if t_idx == -1:
            break
            
        candidates = target_to_blocks[t_idx]
        best_block = -1
        max_new = -1
        
        for b in candidates:
            if x[b]:
                continue
            new_cov_count = 0
            for s in block_subsets[b]:
                if cov[s] == 0:
                    new_cov_count += 1
            if new_cov_count > max_new:
                max_new = new_cov_count
                best_block = b
                
        if best_block != -1:
            add_block(best_block)
            
    # Remove redundant blocks
    removed = True
    while removed:
        removed = False
        for i in selected_indices[:]:
            redundant = True
            for s in block_subsets[i]:
                if cov[s] <= 1:
                    redundant = False
                    break
            if redundant:
                remove_block(i)
                removed = True

    current_best_count = selected_count
    best_solution = x[:]
    
    start_time = time.time()
    time_limit = 110.0
    
    # Simulated Annealing Parameters
    T = 50.0
    cooling_rate = 0.99995
    min_T = 0.01
    penalty = 1000
    
    def get_cost():
        return selected_count + penalty * uncovered_count

    current_cost = get_cost()
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Generate Move
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a random selected block
            if not selected_indices:
                continue
            i = random.choice(selected_indices)
            remove_block(i)
            new_cost = get_cost()
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if selected_count < current_best_count and uncovered_count == 0:
                    current_best_count = selected_count
                    best_solution = x[:]
            else:
                add_block(i) # Revert
        elif move_type < 0.7:
            # Add a random unselected block
            i = random.randint(0, N - 1)
            if x[i]:
                continue
            add_block(i)
            new_cost = get_cost()
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if selected_count < current_best_count and uncovered_count == 0:
                    current_best_count = selected_count
                    best_solution = x[:]
            else:
                remove_block(i) # Revert
        else:
            # Swap: Remove one selected, Add one unselected
            if not selected_indices:
                continue
            i = random.choice(selected_indices)
            j = random.randint(0, N - 1)
            if j == i or x[j]:
                continue
            
            remove_block(i)
            add_block(j)
            
            new_cost = get_cost()
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if selected_count < current_best_count and uncovered_count == 0:
                    current_best_count = selected_count
                    best_solution = x[:]
            else:
                # Revert
                remove_block(j)
                add_block(i)
                
        T *= cooling_rate
        
    # Final Output
    blocks = []
    for i in range(N):
        if best_solution[i]:
            blocks.append(list(all_blocks[i]))
            
    return blocks
