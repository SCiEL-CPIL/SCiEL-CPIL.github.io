import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute t-sets
    t_sets = list(itertools.combinations(range(v), t))
    n_t_sets = len(t_sets)
    t_set_to_idx = {ts: i for i, ts in enumerate(t_sets)}
    
    # Precompute k-sets (blocks)
    k_sets = list(itertools.combinations(range(v), k))
    n_k_sets = len(k_sets)
    
    # Mapping: block index -> list of t-set indices
    block_to_tsets = []
    # Mapping: t-set index -> list of block indices
    tset_to_blocks = [[] for _ in range(n_t_sets)]
    
    for b_idx, block in enumerate(k_sets):
        block_tsets = []
        for ts in itertools.combinations(block, t):
            idx = t_set_to_idx[ts]
            block_tsets.append(idx)
            tset_to_blocks[idx].append(b_idx)
        block_to_tsets.append(block_tsets)
        
    counts = [0] * n_t_sets
    current_blocks = []
    uncovered_count = n_t_sets
    
    def add_block(b_idx):
        nonlocal uncovered_count
        for ts_idx in block_to_tsets[b_idx]:
            if counts[ts_idx] == 0:
                uncovered_count -= 1
            counts[ts_idx] += 1
        current_blocks.append(b_idx)
        
    # Greedy Initialization
    while uncovered_count > 0 and time.time() < deadline:
        # Find an uncovered t-set
        # Randomize start to avoid bias
        start_scan = random.randint(0, n_t_sets - 1)
        target_ts_idx = -1
        for i in range(n_t_sets):
            idx = (start_scan + i) % n_t_sets
            if counts[idx] == 0:
                target_ts_idx = idx
                break
                
        if target_ts_idx == -1:
            break
            
        best_block_idx = -1
        best_gain = 0
        # Candidates are blocks covering target_ts_idx
        for b_idx in tset_to_blocks[target_ts_idx]:
            gain = 0
            for ts_idx in block_to_tsets[b_idx]:
                if counts[ts_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx != -1:
            add_block(best_block_idx)
            
    # Optimization Phase
    # Local search: try to remove blocks, swap if necessary
    while time.time() < deadline:
        random.shuffle(current_blocks)
        
        i = 0
        while i < len(current_blocks):
            if time.time() > deadline:
                break
            
            b_idx = current_blocks[i]
            
            # Tentatively remove b_idx
            removed_uncovered = []
            for ts_idx in block_to_tsets[b_idx]:
                counts[ts_idx] -= 1
                if counts[ts_idx] == 0:
                    removed_uncovered.append(ts_idx)
                    
            if not removed_uncovered:
                # Successfully removed
                current_blocks.pop(i)
                # Do not increment i
                continue
            else:
                # Need to find replacement
                best_repl_idx = -1
                best_repl_gain = 0
                
                # Pick a target from removed_uncovered
                target_ts = removed_uncovered[0]
                rem_set = set(removed_uncovered)
                
                for cand_idx in tset_to_blocks[target_ts]:
                    gain = 0
                    for ts_idx in block_to_tsets[cand_idx]:
                        if ts_idx in rem_set:
                            gain += 1
                    if gain > best_repl_gain:
                        best_repl_gain = gain
                        best_repl_idx = cand_idx
                        
                if best_repl_idx != -1 and best_repl_gain > 0:
                    # Swap: add best_repl_idx
                    for ts_idx in block_to_tsets[best_repl_idx]:
                        counts[ts_idx] += 1
                    current_blocks[i] = best_repl_idx
                    i += 1
                else:
                    # Restore b_idx
                    for ts_idx in block_to_tsets[b_idx]:
                        counts[ts_idx] += 1
                    i += 1
                    
    result = [list(k_sets[b_idx]) for b_idx in current_blocks]
    return result
