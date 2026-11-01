import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 140.0
    
    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    masks = [0] * n_k
    for b, k_sub in enumerate(k_subsets):
        m = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_subset_to_idx[t_sub]
            m |= (1 << idx)
        masks[b] = m
        
    # Greedy Initialization
    uncovered = (1 << n_t) - 1
    selected_indices = []
    
    while uncovered:
        best_gain = -1
        best_b = -1
        for b in range(n_k):
            gain = bin(masks[b] & uncovered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_b = b
        if best_b == -1:
            break
        selected_indices.append(best_b)
        uncovered &= ~masks[best_b]
        
        if time.time() - start_time > timeout:
            break

    selected = selected_indices
    
    def remove_redundant(selected):
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                union_others = 0
                for j in range(len(selected)):
                    if i != j:
                        union_others |= masks[selected[j]]
                if (masks[selected[i]] & ~union_others) == 0:
                    selected.pop(i)
                    changed = True
                    break
        return selected

    def try_2_for_1_swap(selected):
        n_sel = len(selected)
        if n_sel < 2:
            return selected, False
            
        improved = False
        for i in range(n_sel):
            if time.time() - start_time > timeout:
                break
            for j in range(i + 1, n_sel):
                if time.time() - start_time > timeout:
                    break
                    
                union_others = 0
                for m_idx in range(n_sel):
                    if m_idx != i and m_idx != j:
                        union_others |= masks[selected[m_idx]]
                        
                required = (masks[selected[i]] | masks[selected[j]]) & ~union_others
                
                if required == 0:
                    selected.pop(j)
                    improved = True
                    break
                else:
                    # Check if possible
                    if bin(required).count('1') > 35:
                        continue
                        
                    found = -1
                    for m in range(n_k):
                        if (required & ~masks[m]) == 0:
                            found = m
                            break
                            
                    if found != -1:
                        selected.pop(j)
                        selected.pop(i)
                        selected.append(found)
                        improved = True
                        break
                        
            if improved:
                break
                
        return selected, improved

    selected = remove_redundant(selected)
    
    while time.time() - start_time < timeout:
        selected = remove_redundant(selected)
        new_selected, improved = try_2_for_1_swap(selected)
        if improved:
            selected = new_selected
        else:
            break

    result = [list(k_subsets[i]) for i in selected]
    return result
