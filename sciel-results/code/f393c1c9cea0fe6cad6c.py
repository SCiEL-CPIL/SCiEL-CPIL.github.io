import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    edge_map = {}
    
    # Generate all edges
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = len(edges) - 1
    
    n_edges = len(edges)
    
    # Precompute for each edge, the list of "opposite" edge pairs in each C4 it belongs to.
    # If edge e is in a C4 with edges a, b, c (where e is one of the 4), then adding e
    # creates a C4 iff both of the other two specific edges are already present.
    # More precisely: for edge e (flipping bit i between u and v), for each other bit j,
    # the C4 has vertices u, v, u^(1<<j), v^(1<<j).
    # The 4 edges are:
    #   e = (u, v) [bit i]
    #   e_j1 = (u, u^(1<<j)) [bit j]
    #   e_j2 = (v, v^(1<<j)) [bit j]
    #   e_opp = (u^(1<<j), v^(1<<j)) [bit i]
    # A C4 is formed if all 4 are present. So if we're considering adding e,
    # a C4 is formed iff e_j1, e_j2, and e_opp are all in the selected set.
    # But actually, we can precompute: for each edge e, and each C4 it's in,
    # the other 3 edges. A C4 is complete iff all 4 are selected.
    # So when checking if we can add e, we check: for each C4 containing e,
    # are the other 3 edges all selected? If yes, we cannot add e.
    
    # Let's precompute: for each edge index, a list of tuples of 3 edge indices
    # (the other 3 edges in each C4 that this edge belongs to)
    c4_conflicts = [[] for _ in range(n_edges)]
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_map[e1]
            idx2 = edge_map[e2]
            idx3 = edge_map[e3]
            
            c4_conflicts[idx].append((idx1, idx2, idx3))
    
    # Use a boolean array for fast membership checks
    def is_valid_add(edge_idx, selected_arr):
        for t in c4_conflicts[edge_idx]:
            if selected_arr[t[0]] and selected_arr[t[1]] and selected_arr[t[2]]:
                return False
        return True
    
    def greedy_fill(selected_arr, available_indices, random_order=True):
        if random_order:
            shuffled = list(available_indices)
            random.shuffle(shuffled)
            order = shuffled
        else:
            order = available_indices
        for idx in order:
            if not selected_arr[idx]:
                if is_valid_add(idx, selected_arr):
                    selected_arr[idx] = True
    
    def count_edges(selected_arr):
        return int(np.sum(selected_arr))
    
    def local_search_improve(selected_arr, max_iterations=500):
        """
        Improve by trying to remove one edge and add multiple others.
        """
        improved = True
        iterations = 0
        n_selected = count_edges(selected_arr)
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try removing each selected edge and greedily re-filling
            selected_indices = np.where(selected_arr)[0]
            
            for remove_idx in selected_indices:
                if n_selected <= 1:
                    break
                    
                # Save state
                saved = selected_arr.copy()
                
                # Remove the edge
                selected_arr[remove_idx] = False
                
                # Get all unselected edges
                remaining = np.where(~selected_arr)[0]
                
                # Try a few random orders
                for attempt in range(2):
                    test_arr = saved.copy()
                    test_arr[remove_idx] = False
                    rem = list(remaining)
                    random.shuffle(rem)
                    
                    for idx in rem:
                        if not test_arr[idx]:
                            if is_valid_add(idx, test_arr):
                                test_arr[idx] = True
                    
                    new_count = count_edges(test_arr)
                    if new_count > n_selected:
                        selected_arr = test_arr
                        n_selected = new_count
                        improved = True
                        break
        
        return selected_arr, n_selected
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 150s limit
    
    random.seed(42)
    all_indices = list(range(n_edges))
    
    best_arr = np.zeros(n_edges, dtype=bool)
    best_size = 0
    
    # Multiple restarts with greedy + local search
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Random initial greedy
        current_arr = np.zeros(n_edges, dtype=bool)
        shuffled = list(all_indices)
        random.shuffle(shuffled)
        greedy_fill(current_arr, shuffled)
        
        current_size = count_edges(current_arr)
        
        # Local search improvement
        current_arr, current_size = local_search_improve(current_arr, max_iterations=100)
        
        if current_size > best_size:
            best_size = current_size
            best_arr = current_arr.copy()
    
    # Final polish on the best solution with more iterations
    if time.time() - start_time < time_limit - 10:
        best_arr, best_size = local_search_improve(best_arr, max_iterations=200)
    
    # Convert best_arr to list of [u, v]
    result = []
    for idx in np.where(best_arr)[0]:
        result.append(list(edges[idx]))
        
    return result
