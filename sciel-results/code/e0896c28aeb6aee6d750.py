import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all t-subsets and map to index
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {s: i for i, s in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # Precompute all k-subsets (blocks) and the t-subsets they cover
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    block_to_t_subs = []
    for b in blocks:
        t_idx = [t_sub_to_idx[s] for s in itertools.combinations(b, t)]
        block_to_t_subs.append(t_idx)
        
    covered = [0] * num_t_subs
    current_blocks = []
    current_block_set = set()
    uncovered_count = num_t_subs
    
    def add_block(b_idx):
        nonlocal uncovered_count
        current_blocks.append(b_idx)
        current_block_set.add(b_idx)
        for t_idx in block_to_t_subs[b_idx]:
            if covered[t_idx] == 0:
                uncovered_count -= 1
            covered[t_idx] += 1
            
    def remove_block(b_idx):
        nonlocal uncovered_count
        current_blocks.remove(b_idx)
        current_block_set.remove(b_idx)
        for t_idx in block_to_t_subs[b_idx]:
            covered[t_idx] -= 1
            if covered[t_idx] == 0:
                uncovered_count += 1
                
    def restore_state(sol_blocks):
        nonlocal uncovered_count
        current_blocks.clear()
        current_block_set.clear()
        for i in range(num_t_subs):
            covered[i] = 0
        uncovered_count = num_t_subs
        for b_idx in sol_blocks:
            add_block(b_idx)

    # Initial Greedy
    while uncovered_count > 0 and time.time() < deadline:
        best_gain = -1
        best_b = -1
        for b_idx in range(num_blocks):
            if b_idx in current_block_set:
                continue
            gain = 0
            for t_idx in block_to_t_subs[b_idx]:
                if covered[t_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_b = b_idx
                if gain == len(block_to_t_subs[b_idx]):
                    break
        if best_b == -1:
            break
        add_block(best_b)
        
    best_solution = list(current_blocks)
    best_size = len(current_blocks)
    
    def remove_redundant():
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Iterate backwards
            for i in range(len(current_blocks) - 1, -1, -1):
                b_idx = current_blocks[i]
                is_redundant = True
                for t_idx in block_to_t_subs[b_idx]:
                    if covered[t_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    remove_block(b_idx)
                    changed = True
                    
    def optimize():
        remove_redundant()
        
        # Try swaps
        for i in range(len(current_blocks)):
            if time.time() > deadline:
                break
            b_idx = current_blocks[i]
            
            # Tentatively remove
            remove_block(b_idx)
            
            # Identify uncovered
            uncovered_set = set()
            for t_idx in block_to_t_subs[b_idx]:
                if covered[t_idx] == 0:
                    uncovered_set.add(t_idx)
            
            if not uncovered_set:
                continue
                
            # Find best replacement
            best_gain = -1
            best_b = -1
            for nb_idx in range(num_blocks):
                if nb_idx in current_block_set:
                    continue
                gain = 0
                for t_idx in block_to_t_subs[nb_idx]:
                    if t_idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = nb_idx
            
            if best_gain == len(uncovered_set) and best_b != -1:
                add_block(best_b)
            else:
                add_block(b_idx)
                
        remove_redundant()

    while time.time() < deadline:
        optimize()
        
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_solution = list(current_blocks)
            
        # Perturbation
        if len(current_blocks) > 0 and random.random() < 0.3:
            # Save state
            saved_blocks = list(current_blocks)
            
            # Remove a random block
            idx_to_remove = random.randint(0, len(current_blocks) - 1)
            b_idx = current_blocks[idx_to_remove]
            remove_block(b_idx)
            
            # Repair greedily
            while uncovered_count > 0 and time.time() < deadline:
                best_gain = -1
                best_b = -1
                for nb_idx in range(num_blocks):
                    if nb_idx in current_block_set:
                        continue
                    gain = 0
                    for t_idx in block_to_t_subs[nb_idx]:
                        if covered[t_idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_b = nb_idx
                if best_b == -1:
                    break
                add_block(best_b)
            
            optimize()
            
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_solution = list(current_blocks)
            else:
                # Revert to saved state if not improved
                restore_state(saved_blocks)

    result = [list(blocks[b_idx]) for b_idx in best_solution]
    return result
