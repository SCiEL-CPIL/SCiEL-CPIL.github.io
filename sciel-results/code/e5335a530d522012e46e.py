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
        
    # Precompute target_to_blocks for initialization
    target_to_blocks = [[] for _ in range(M)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            target_to_blocks[s].append(i)

    random.seed(42)
    
    # State variables
    x = [0] * N  # 1 if selected
    cov = [0] * M # Coverage count for each target
    
    LAMBDA = N + 1
    
    num_uncovered = M
    num_selected = 0
    current_cost = LAMBDA * M
    
    def add_block(i):
        nonlocal num_uncovered, num_selected, current_cost
        if x[i]:
            return
        x[i] = 1
        num_selected += 1
        delta_cost = 1
        for s in block_subsets[i]:
            if cov[s] == 0:
                num_uncovered -= 1
                delta_cost -= LAMBDA
            cov[s] += 1
        current_cost += delta_cost
        
    def remove_block(i):
        nonlocal num_uncovered, num_selected, current_cost
        if not x[i]:
            return
        x[i] = 0
        num_selected -= 1
        delta_cost = -1
        for s in block_subsets[i]:
            cov[s] -= 1
            if cov[s] == 0:
                num_uncovered += 1
                delta_cost += LAMBDA
        current_cost += delta_cost

    # 1. Initialization: Fast Greedy
    uncovered_indices = list(range(M))
    while uncovered_indices:
        t_idx = random.choice(uncovered_indices)
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
            uncovered_indices = [i for i in uncovered_indices if cov[i] == 0]
        else:
            break

    best_cost = current_cost
    best_x = x.copy()
    
    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 110.0
    total_time = time_limit
    
    T_start = 50.0
    T_end = 0.5
    
    while time.time() - start_time < time_limit:
        time_remaining = max(0, time_limit - (time.time() - start_time))
        T = T_start * (time_remaining / total_time)
        if T < T_end:
            T = T_end
            
        # Generate move: Pick a random block to toggle
        i = random.randint(0, N - 1)
        
        if x[i]:
            # Try to remove
            old_cost = current_cost
            remove_block(i)
            delta = current_cost - old_cost
            
            # Metropolis criterion
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                if num_uncovered == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
            else:
                # Reject: revert
                add_block(i)
        else:
            # Try to add
            old_cost = current_cost
            add_block(i)
            delta = current_cost - old_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                if num_uncovered == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
            else:
                # Reject: revert
                remove_block(i)
                    
    x = best_x
    selected = [i for i in range(N) if x[i]]
    
    blocks = []
    for i in selected:
        blocks.append(list(all_blocks[i]))
        
    unique_blocks = []
    seen = set()
    for b in blocks:
        b.sort()
        t_b = tuple(b)
        if t_b not in seen:
            seen.add(t_b)
            unique_blocks.append(b)
            
    return unique_blocks
