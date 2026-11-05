import time
import random
import numpy as np

def construct():
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Map (i, j) to index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - (i * (i + 1) // 2) + (j - i - 1)
        
    # Precompute K4 edge tuples
    k4_edges = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    edges = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
                    ]
                    k4_edges.append(edges)
                    
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)
    num_k4s = len(k4_edges)
    
    def fallback():
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]

    def check_vertex_labeling(labels, color_func, k):
        """Check if a vertex labeling produces a valid edge coloring."""
        colors = np.zeros(num_edges, dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                colors[edge_idx(i, j)] = color_func(labels[i], labels[j]) % k
        
        # Check all K4s
        for edges in k4_edges:
            k4_colors = set(colors[edges])
            if len(k4_colors) < 5:
                return False
        return True

    def vertex_local_search(num_labels, color_func, k, time_limit):
        """Local search on vertex labels."""
        start_time = time.time()
        best_labels = None
        best_cost = float('inf')
        
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            labels = np.random.randint(0, num_labels, size=n, dtype=np.int32)
            
            # Compute cost: number of K4s with < 5 distinct colors
            def compute_cost(lbls):
                colors = np.zeros(num_edges, dtype=np.int32)
                for i in range(n):
                    for j in range(i + 1, n):
                        colors[edge_idx(i, j)] = color_func(lbls[i], lbls[j]) % k
                
                cost = 0
                for edges in k4_edges:
                    k4_colors = set(colors[edges])
                    if len(k4_colors) < 5:
                        cost += (5 - len(k4_colors)) ** 2
                return cost
            
            cur_cost = compute_cost(labels)
            if cur_cost == 0:
                # Found valid solution
                colors = np.zeros(num_edges, dtype=np.int32)
                for i in range(n):
                    for j in range(i + 1, n):
                        colors[edge_idx(i, j)] = color_func(labels[i], labels[j]) % k
                return colors.tolist()
            
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_labels = labels.copy()
            
            no_improve = 0
            while time.time() - start_time < time_limit:
                if cur_cost == 0:
                    break
                    
                # Pick a random vertex to change
                v = random.randint(0, n - 1)
                old_label = labels[v]
                
                best_new_label = old_label
                best_delta = 0
                
                for new_label in range(num_labels):
                    if new_label == old_label:
                        continue
                    
                    labels[v] = new_label
                    new_cost = compute_cost(labels)
                    delta = new_cost - cur_cost
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_label = new_label
                
                labels[v] = best_new_label
                if best_delta < 0:
                    cur_cost += best_delta
                    no_improve = 0
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_labels = labels.copy()
                else:
                    no_improve += 1
                    if no_improve > 20:
                        # Random kick
                        for _ in range(3):
                            kick_v = random.randint(0, n - 1)
                            labels[kick_v] = random.randint(0, num_labels - 1)
                        cur_cost = compute_cost(labels)
                        no_improve = 0
                        if cur_cost < best_cost:
                            best_cost = cur_cost
                            best_labels = labels.copy()
        
        if best_labels is not None and best_cost == 0:
            colors = np.zeros(num_edges, dtype=np.int32)
            for i in range(n):
                for j in range(i + 1, n):
                    colors[edge_idx(i, j)] = color_func(best_labels[i], best_labels[j]) % k
            return colors.tolist()
        return None

    # Try different color functions and label counts
    time_start = time.time()
    total_time_limit = 100.0
    
    # Color function 1: (a * b) % k
    def color_func_mult(a, b):
        return (a * b) % 17
    
    # Color function 2: (a + b) % k  
    def color_func_sum(a, b):
        return (a + b) % 17
    
    # Color function 3: (a * b + a + b) % k
    def color_func_mix(a, b):
        return (a * b + a + b) % 17

    # Try with 2, 3, 4, 5, 6 labels
    for num_labels in [2, 3, 4, 5, 6]:
        if time.time() - time_start > total_time_limit:
            break
        
        time_per_config = min(20.0, total_time_limit - (time.time() - time_start))
        
        for func_name, func in [("mult", color_func_mult), ("sum", color_func_sum), ("mix", color_func_mix)]:
            if time.time() - time_start > total_time_limit:
                break
            
            res = vertex_local_search(num_labels, func, 17, time_per_config / 3)
            if res is not None:
                # Verify
                colors = np.array(res, dtype=np.int32)
                valid = True
                for edges in k4_edges:
                    if len(set(colors[edges])) < 5:
                        valid = False
                        break
                if valid:
                    # Count distinct colors
                    num_colors = len(set(colors))
                    return res
    
    # If vertex labeling didn't work, fall back to direct edge search
    if time.time() - time_start < total_time_limit:
        # Direct edge local search (simplified)
        def edge_local_search(k, time_limit):
            start_time = time.time()
            best_solution = None
            best_cost = float('inf')
            
            for restart in range(10):
                if time.time() - start_time > time_limit:
                    break
                
                colors = np.random.randint(0, k, size=num_edges, dtype=np.int32)
                
                def compute_cost(cols):
                    cost = 0
                    for edges in k4_edges:
                        k4_colors = set(cols[edges])
                        if len(k4_colors) < 5:
                            cost += (5 - len(k4_colors)) ** 2
                    return cost
                
                cur_cost = compute_cost(colors)
                if cur_cost == 0:
                    return colors.tolist()
                
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_solution = colors.copy()
                
                no_improve = 0
                while time.time() - start_time < time_limit:
                    if cur_cost == 0:
                        break
                    
                    # Pick random edge
                    e_idx = random.randint(0, num_edges - 1)
                    old_c = colors[e_idx]
                    
                    best_new_c = old_c
                    best_delta = 0
                    
                    for new_c in range(k):
                        if new_c == old_c:
                            continue
                        colors[e_idx] = new_c
                        new_cost = compute_cost(colors)
                        delta = new_cost - cur_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_new_c = new_c
                    
                    colors[e_idx] = best_new_c
                    if best_delta < 0:
                        cur_cost += best_delta
                        no_improve = 0
                        if cur_cost < best_cost:
                            best_cost = cur_cost
                            best_solution = colors.copy()
                    else:
                        no_improve += 1
                        if no_improve > 50:
                            for _ in range(5):
                                kick_e = random.randint(0, num_edges - 1)
                                colors[kick_e] = random.randint(0, k - 1)
                            cur_cost = compute_cost(colors)
                            no_improve = 0
                            if cur_cost < best_cost:
                                best_cost = cur_cost
                                best_solution = colors.copy()
            
            if best_solution is not None and best_cost == 0:
                return best_solution.tolist()
            return None
        
        res = edge_local_search(13, 30.0)
        if res is not None:
            return res
    
    return fallback()
