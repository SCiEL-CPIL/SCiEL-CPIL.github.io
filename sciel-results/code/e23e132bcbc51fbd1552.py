import itertools
import random
import time

def construct(v, k, t):
    all_blocks = list(itertools.combinations(range(v), k))
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    num_blocks = len(all_blocks)
    num_triples = len(all_triples)
    
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(ts)
        
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_to_blocks[t_idx].append(i)
            
    start_time = time.time()
    max_time = 115
    
    global_best_solution = None
    global_best_size = float('inf')
    
    def run_local_search(seed):
        nonlocal global_best_solution, global_best_size
        random.seed(seed)
        
        selected = [False] * num_blocks
        selected_list = []
        triple_cover_count = [0] * num_triples
        num_uncovered = num_triples
        
        def add_block(b_idx):
            nonlocal num_uncovered
            if selected[b_idx]:
                return
            selected[b_idx] = True
            selected_list.append(b_idx)
            for t_idx in block_triples[b_idx]:
                if triple_cover_count[t_idx] == 0:
                    num_uncovered -= 1
                triple_cover_count[t_idx] += 1
                
        def remove_block(b_idx):
            nonlocal num_uncovered
            if not selected[b_idx]:
                return
            selected[b_idx] = False
            # Remove from selected_list
            # Find index
            # Since list is small, linear search is fine
            for j in range(len(selected_list)):
                if selected_list[j] == b_idx:
                    selected_list[j] = selected_list[-1]
                    selected_list.pop()
                    break
            
            for t_idx in block_triples[b_idx]:
                triple_cover_count[t_idx] -= 1
                if triple_cover_count[t_idx] == 0:
                    num_uncovered += 1

        def get_best_block_for_triple(t_idx):
            best_block = -1
            best_gain = -1
            candidates = []
            
            for b_idx in triple_to_blocks[t_idx]:
                if selected[b_idx]:
                    continue
                gain = 0
                for t2 in block_triples[b_idx]:
                    if triple_cover_count[t2] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    candidates = [b_idx]
                elif gain == best_gain:
                    candidates.append(b_idx)
                    
            if not candidates:
                return -1
            return random.choice(candidates)

        # Main Loop
        while time.time() - start_time < max_time:
            if num_uncovered > 0:
                # Find uncovered triples
                # Optimization: Keep a list of uncovered triples?
                # Updating it is hard. Scanning is 2024 checks.
                uncovered_triples = []
                for i in range(num_triples):
                    if triple_cover_count[i] == 0:
                        uncovered_triples.append(i)
                
                if not uncovered_triples:
                    break # Should be caught by num_uncovered, but safe
                    
                t_idx = random.choice(uncovered_triples)
                b_idx = get_best_block_for_triple(t_idx)
                if b_idx != -1:
                    add_block(b_idx)
                else:
                    break
            else:
                # Minimize blocks
                removed_any = False
                attempts = 0
                # Try to remove blocks
                # Limit attempts to avoid infinite loop if stuck
                max_attempts = max(20, len(selected_list))
                
                while attempts < max_attempts and selected_list:
                    b_idx = random.choice(selected_list)
                    removable = True
                    for t_idx in block_triples[b_idx]:
                        if triple_cover_count[t_idx] <= 1:
                            removable = False
                            break
                    if removable:
                        remove_block(b_idx)
                        removed_any = True
                        attempts = 0
                    else:
                        attempts += 1
                        
                if not removed_any:
                    # Local minimum, perform random walk
                    if selected_list:
                        b_idx = random.choice(selected_list)
                        remove_block(b_idx)
                    else:
                        break
                        
            # Update global best
            if num_uncovered == 0:
                current_size = len(selected_list)
                if current_size < global_best_size:
                    global_best_size = current_size
                    global_best_solution = selected_list.copy()

    # Run multiple restarts
    for i in range(40):
        if time.time() - start_time > 110:
            break
        run_local_search(i)
        
    if global_best_solution is None:
        # Fallback
        return []
        
    return [list(all_blocks[i]) for i in global_best_solution]
