import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    d = 7
    n = 1 << d  # 128
    total_edges = d * (1 << (d - 1))  # 448
    
    # Build all edges
    all_edges = []  # (u, v, dim)
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))
    
    num_edges = len(all_edges)  # 448
    
    # Precompute blocking triads for each edge
    blocking = [[] for _ in range(num_edges)]
    for idx, (x, y, i) in enumerate(all_edges):
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = x ^ (1 << i) ^ (1 << j)
            e1 = (min(x, c), max(x, c))
            e2 = (min(x ^ (1 << i), e), max(x ^ (1 << i), e))
            e3 = (min(c, e), max(c, e))
            blocking[idx].append((e1, e2, e3))
    
    def can_add(chosen_set, edge_idx):
        for triad in blocking[edge_idx]:
            if all(t in chosen_set for t in triad):
                return False
        return True
    
    def greedy(order):
        chosen_set = set()
        result = []
        for edge_idx in order:
            if can_add(chosen_set, edge_idx):
                x, y, _ = all_edges[edge_idx]
                chosen_set.add((x, y))
                result.append(edge_idx)
        return result
    
    def to_edge_list(idx_list):
        return [[all_edges[i][0], all_edges[i][1]] for i in idx_list]
    
    def local_search(idx_list, chosen_set, max_time=None):
        if max_time and time.time() > max_time:
            return idx_list, chosen_set
        
        missing = [i for i in range(num_edges) if i not in set(idx_list)]
        
        # Try to add missing edges
        changed = True
        while changed:
            changed = False
            if max_time and time.time() > max_time:
                break
            for i in missing:
                if max_time and time.time() > max_time:
                    break
                if can_add(chosen_set, i):
                    x, y, _ = all_edges[i]
                    chosen_set.add((x, y))
                    idx_list.append(i)
                    missing.remove(i)
                    changed = True
                    break
        
        # Try swap moves
        if max_time and time.time() < max_time:
            for remove_idx in range(len(idx_list)):
                if max_time and time.time() > max_time:
                    break
                rem_i = idx_list[remove_idx]
                rem_x, rem_y, _ = all_edges[rem_i]
                rem_tuple = (rem_x, rem_y)
                
                for add_i in missing:
                    if max_time and time.time() > max_time:
                        break
                    test_set = set(chosen_set)
                    test_set.discard(rem_tuple)
                    if can_add(test_set, add_i):
                        chosen_set.discard(rem_tuple)
                        add_x, add_y, _ = all_edges[add_i]
                        chosen_set.add((add_x, add_y))
                        idx_list[remove_idx] = add_i
                        missing.remove(add_i)
                        missing.append(rem_i)
                        break
                else:
                    continue
                break
        
        return idx_list, chosen_set
    
    # Multi-start
    best_idx_list = None
    best_count = 0
    
    # Sequential greedy
    seq_order = list(range(num_edges))
    result = greedy(seq_order)
    best_count = len(result)
    best_idx_list = result
    
    # Random restarts
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline:
            break
        
        order = list(range(num_edges))
        random.shuffle(order)
        result = greedy(order)
        count = len(result)
        
        if count > best_count:
            best_count = count
            best_idx_list = result
        
        # Local search on promising results
        if count >= best_count - 3:
            chosen_set = set((all_edges[i][0], all_edges[i][1]) for i in result)
            improved_list, improved_set = local_search(result[:], chosen_set, max_time=deadline)
            if len(improved_list) > best_count:
                best_count = len(improved_list)
                best_idx_list = improved_list
    
    return to_edge_list(best_idx_list)
