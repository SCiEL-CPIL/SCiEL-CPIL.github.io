import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(v))
    t_subsets = list(itertools.combinations(points, t))
    t_subset_map = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(points, k))
    num_k_subsets = len(k_subsets)
    
    k_to_t = []
    for ks in k_subsets:
        t_list = [t_subset_map[ts] for ts in itertools.combinations(ks, t)]
        k_to_t.append(t_list)
        
    t_to_k = [[] for _ in range(num_t_subsets)]
    for k_idx, t_list in enumerate(k_to_t):
        for t_idx in t_list:
            t_to_k[t_idx].append(k_idx)
            
    best_solution = None
    best_len = float('inf')
    
    def greedy_with_deletion():
        is_uncovered = np.ones(num_t_subsets, dtype=bool)
        uncovered_list = list(range(num_t_subsets))
        blocks = []
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        while uncovered_list:
            # Pick random index from uncovered_list
            idx = random.randint(0, len(uncovered_list) - 1)
            s = uncovered_list[idx]
            
            # Ensure s is actually uncovered (it should be, but just in case)
            if not is_uncovered[s]:
                # Swap with last and pop
                uncovered_list[idx] = uncovered_list[-1]
                uncovered_list.pop()
                continue
            
            candidates = t_to_k[s]
            best_candidate = -1
            best_gain = -1
            
            for c in candidates:
                gain = 0
                for t_idx in k_to_t[c]:
                    if is_uncovered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c
            
            blocks.append(best_candidate)
            
            for t_idx in k_to_t[best_candidate]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # We don't remove from uncovered_list immediately to avoid O(N) cost
                    # We'll clean up later or just let the loop handle stale entries
                    # But we need to ensure we don't pick stale entries often.
                    # Actually, the swap-pop above handles it if we encounter it.
                    # But we also need to remove it from the list if we want the list to shrink.
                    # For now, let's just mark it false. The loop condition `while uncovered_list`
                    # will keep running until the list is empty.
                    # We must remove it from the list to terminate.
                    # Let's do the swap-pop here too?
                    # No, `t_idx` is not at `idx`.
                    # We can't easily remove arbitrary elements from a list in O(1).
                    # So `uncovered_list` will contain stale entries.
                    # The check `if not is_uncovered[s]` handles this.
                    # The list will shrink as we encounter stale entries.
                    pass
                cover_count[t_idx] += 1
                
            # To speed up termination, we can periodically filter the list
            # Or just let it run. The number of stale entries is bounded by total steps * 35.
            # Total steps ~150. Stale entries ~5000. List size starts at 3876.
            # It will grow? No, we only add stale entries if we don't remove them.
            # We don't add new entries. We just mark existing ones as false.
            # So the list size stays constant until we pop.
            # We pop when we pick a stale one.
            # So it will eventually empty.
            
        # Deletion Phase
        random.shuffle(blocks)
        new_blocks = []
        for b in blocks:
            redundant = True
            for t_idx in k_to_t[b]:
                if cover_count[t_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                for t_idx in k_to_t[b]:
                    cover_count[t_idx] -= 1
            else:
                new_blocks.append(b)
        
        return new_blocks

    while time.time() < deadline:
        solution = greedy_with_deletion()
        if len(solution) < best_len:
            best_len = len(solution)
            best_solution = solution
            
        if time.time() > deadline - 2:
            break
            
    if best_solution is None:
        return [list(k_subsets[0])]
        
    return [list(k_subsets[b]) for b in best_solution]
