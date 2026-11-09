import random
import time

def construct():
    d = 7
    n = 2 ** d
    total_edges = d * (n // 2)
    
    # Precompute all edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v))
    
    # Helper to check if adding edge (u, v) creates a 4-cycle
    def is_safe(u, v, selected):
        mask = u ^ v
        i = mask.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            mask_j = 1 << j
            u_j = u ^ mask_j
            v_j = v ^ mask_j
            
            e1 = (min(u, u_j), max(u, u_j))
            e2 = (min(v, v_j), max(v, v_j))
            e3 = (min(u_j, v_j), max(u_j, v_j))
            
            if e1 in selected and e2 in selected and e3 in selected:
                return False
        return True

    def greedy_fill(selected, current_edges):
        # Try to add any edge not in selected that is safe.
        # We iterate through a shuffled list of unselected edges to randomize the greedy choice.
        unselected = [e for e in all_edges if e not in selected]
        random.shuffle(unselected)
        
        for u, v in unselected:
            if is_safe(u, v, selected):
                selected.add((u, v))
                current_edges.append((u, v))
        return selected, current_edges

    best_count = 0
    best_edges = []
    
    start_time = time.time()
    max_time = 110.0
    
    # Initial construction: Run a few random greedy trials to find a good starting point
    for _ in range(20):
        selected = set()
        current_edges = []
        greedy_fill(selected, current_edges)
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
    if best_count >= 308:
        return [list(e) for e in best_edges]

    # Local Search: Modify the current best solution and refill
    current_selected = set(best_edges)
    current_edges = list(best_edges)
    
    while time.time() - start_time < max_time:
        # Randomly remove a small number of edges to allow reconfiguration
        num_remove = random.randint(1, 5)
        
        if len(current_edges) > num_remove:
            remove_indices = random.sample(range(len(current_edges)), num_remove)
            to_remove = set()
            for idx in remove_indices:
                to_remove.add(current_edges[idx])
            
            for e in to_remove:
                current_selected.remove(e)
                current_edges.remove(e)
        else:
            # If the set is too small, reset it
            current_selected = set()
            current_edges = []
        
        # Refill the gaps
        current_selected, current_edges = greedy_fill(current_selected, current_edges)
        
        # Update best if improved
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
            if best_count >= 308:
                break

    return [list(e) for e in best_edges]
