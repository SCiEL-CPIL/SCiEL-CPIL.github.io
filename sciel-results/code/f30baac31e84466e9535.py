import itertools
import random
import time

_PRECOMP_CACHE = {}

def construct(v, k, t):
    key = (v, k, t)
    if key not in _PRECOMP_CACHE:
        t_subsets = list(itertools.combinations(range(v), t))
        t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
        num_t = len(t_subsets)
        
        block_combinations = list(itertools.combinations(range(v), k))
        num_blocks = len(block_combinations)
        
        block_tsub_indices = [[] for _ in range(num_blocks)]
        for i, block in enumerate(block_combinations):
            for ts in itertools.combinations(block, t):
                idx = t_to_idx[ts]
                block_tsub_indices[i].append(idx)
        
        tsub_blocks = [[] for _ in range(num_t)]
        for i in range(num_blocks):
            for t_idx in block_tsub_indices[i]:
                tsub_blocks[t_idx].append(i)
        
        _PRECOMP_CACHE[key] = (t_subsets, t_to_idx, block_combinations, block_tsub_indices, tsub_blocks, num_t, num_blocks)
    
    t_subsets, t_to_idx, block_combinations, block_tsub_indices, tsub_blocks, num_t, num_blocks = _PRECOMP_CACHE[key]
    
    def run_primal_dual(seed):
        random.seed(seed)
        y = [0.0] * num_t
        covered = [False] * num_t
        selected = [False] * num_blocks
        uncovered_count = num_t
        
        block_sum = [0.0] * num_blocks
        block_uncov_count = [len(block_tsub_indices[i]) for i in range(num_blocks)]
        
        start_time = time.time()
        timeout = 100
        
        while uncovered_count > 0:
            if time.time() - start_time > timeout:
                break
            
            min_time = float('inf')
            for i in range(num_blocks):
                if selected[i] or block_uncov_count[i] == 0:
                    continue
                remaining = 1.0 - block_sum[i]
                if remaining <= 1e-9:
                    min_time = 0.0
                    break
                time_to_tight = remaining / block_uncov_count[i]
                if time_to_tight < min_time:
                    min_time = time_to_tight
            
            if min_time == float('inf'):
                break
            
            for t_idx in range(num_t):
                if not covered[t_idx]:
                    y[t_idx] += min_time
            
            newly_tight = []
            for i in range(num_blocks):
                if selected[i] or block_uncov_count[i] == 0:
                    continue
                block_sum[i] += min_time * block_uncov_count[i]
                if block_sum[i] >= 1.0 - 1e-9:
                    newly_tight.append(i)
            
            random.shuffle(newly_tight)
            
            for i in newly_tight:
                selected[i] = True
                for t_idx in block_tsub_indices[i]:
                    if not covered[t_idx]:
                        covered[t_idx] = True
                        uncovered_count -= 1
                        for j in tsub_blocks[t_idx]:
                            if not selected[j]:
                                block_sum[j] -= y[t_idx]
                                block_uncov_count[j] -= 1
        
        # Fallback greedy if not complete
        if uncovered_count > 0:
            uncovered_set = set(i for i in range(num_t) if not covered[i])
            for _ in range(10000):
                if not uncovered_set:
                    break
                best_i = -1
                best_gain = 0
                for i in range(num_blocks):
                    if selected[i]:
                        continue
                    gain = sum(1 for t_idx in block_tsub_indices[i] if t_idx in uncovered_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
                if best_i == -1 or best_gain == 0:
                    break
                selected[best_i] = True
                for t_idx in block_tsub_indices[best_i]:
                    if t_idx in uncovered_set:
                        uncovered_set.discard(t_idx)
        
        sol_indices = [i for i in range(num_blocks) if selected[i]]
        
        # Remove redundant
        if sol_indices:
            freq = [0] * num_t
            for i in sol_indices:
                for t_idx in block_tsub_indices[i]:
                    freq[t_idx] += 1
            new_sol = []
            for i in sol_indices:
                redundant = True
                for t_idx in block_tsub_indices[i]:
                    if freq[t_idx] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_sol.append(i)
            sol_indices = new_sol
        
        return sol_indices
    
    best_sol = None
    best_len = float('inf')
    
    for seed in range(5):
        sol = run_primal_dual(seed)
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol
    
    if best_sol is None:
        best_sol = []
    
    return [list(block_combinations[i]) for i in best_sol]
