"""Rewritten program: maximum independent set via MILP with branch-and-bound refinement."""
import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 82.0  # safety margin before 90s limit
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # Filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict graph: edge (i,j) if valid_numbers[i] + valid_numbers[j] is a perfect square
    conflict = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b in val_to_idx:
                j = val_to_idx[b]
                if j > i:  # avoid duplicates
                    conflict[i].add(j)
                    conflict[j].add(i)
    
    # Build adjacency bitmasks for fast operations
    adj_bits = []
    for i in range(num_valid):
        bits = 0
        for j in conflict[i]:
            bits |= (1 << j)
        adj_bits.append(bits)
    
    def time_check():
        return time.time() > deadline
    
    # Greedy initial solution: minimum degree first
    def greedy_min_degree():
        remaining = (1 << num_valid) - 1
        selected = []
        while remaining and not time_check():
            min_deg = float('inf')
            best_idx = -1
            temp = remaining
            while temp:
                idx = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = bin(adj_bits[idx] & remaining).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_idx = idx
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining &= ~(1 << best_idx)
            remaining &= ~adj_bits[best_idx]
        return selected
    
    initial = greedy_min_degree()
    best_size = len(initial)
    best_set = set(initial)
    
    # Local search: try to add vertices, then remove to make room
    def local_search(selected_indices):
        selected = set(selected_indices)
        remaining = [(1 << num_valid) - 1]
        remaining_mask = 0
        for i in range(num_valid):
            if i not in selected:
                remaining_mask |= (1 << i)
        
        improved = True
        while improved and not time_check():
            improved = False
            # Try adding each non-selected vertex
            temp = remaining_mask
            while temp:
                idx = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                # Check if idx conflicts with any selected
                if (adj_bits[idx] & (sum((1 << i) for i in selected if i < 64))) == 0:
                    # Check all selected (need full mask)
                    selected_mask = 0
                    for i in selected:
                        selected_mask |= (1 << i)
                    if (adj_bits[idx] & selected_mask) == 0:
                        selected.add(idx)
                        remaining_mask &= ~(1 << idx)
                        remaining_mask &= ~adj_bits[idx]
                        improved = True
                        break
        
        return list(selected), len(selected)
    
    # Run local search on initial
    best_list, best_size = local_search(initial)
    best_set = set(best_list)
    
    # MILP approach using scipy
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csc_matrix
        
        # Objective: maximize sum(x_i)
        c = -np.ones(num_valid)  # minimize negative = maximize positive
        
        # Constraints: for each edge (i,j), x_i + x_j <= 1
        conflicting_pairs = []
        for i in range(num_valid):
            for j in conflict[i]:
                if i < j:
                    conflicting_pairs.append((i, j))
        
        num_constraints = len(conflicting_pairs)
        
        if num_constraints == 0:
            return sorted(valid_numbers)
        
        A = np.zeros((num_constraints, num_valid), dtype=np.float64)
        b = np.ones(num_constraints, dtype=np.float64)
        
        for k, (i, j) in enumerate(conflicting_pairs):
            A[k, i] = 1.0
            A[k, j] = 1.0
        
        lb = np.zeros(num_valid)
        ub = np.ones(num_valid)
        integrality = np.ones(num_valid, dtype=np.int8)
        
        constraints = LinearConstraint(A, -np.inf, b)
        bounds = Bounds(lb, ub)
        
        remaining_time = deadline - time.time()
        if remaining_time < 2:
            return sorted([valid_numbers[i] for i in best_set])
        
        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": remaining_time - 1, "mip_rel_gap": 0.0}
            )
            
            if res.x is not None:
                selected_indices = [i for i in range(num_valid) if res.x[i] > 0.5]
                if len(selected_indices) > best_size:
                    best_size = len(selected_indices)
                    best_set = set(selected_indices)
        except Exception:
            pass
    except ImportError:
        pass
    
    # Branch and bound refinement
    def greedy_color_bound(available_mask):
        """Upper bound using greedy coloring."""
        uncolored = available_mask
        colors = 0
        while uncolored and not time_check():
            colors += 1
            color_class = 0
            remaining = uncolored
            while remaining:
                min_deg = float('inf')
                best_v = -1
                temp = remaining
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    deg = bin(adj_bits[v] & remaining).count('1')
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                if best_v == -1:
                    break
                color_class |= (1 << best_v)
                remaining &= ~(1 << best_v)
                remaining &= ~adj_bits[best_v]
            uncolored &= ~color_class
        return colors
    
    nodes_visited = [0]
    
    def bb_search(available_mask, current_size, current_set):
        if time_check():
            return
        nodes_visited[0] += 1
        
        if available_mask == 0:
            if current_size > best_size:
                nonlocal best_size, best_set
                best_size = current_size
                best_set = set(current_set)
            return
        
        ub = current_size + greedy_color_bound(available_mask)
        if ub <= best_size:
            return
        
        # Pick vertex with max degree in available
        max_deg = -1
        best_v = -1
        temp = available_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            deg = bin(adj_bits[v] & available_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        if best_v == -1:
            if current_size > best_size:
                nonlocal best_size, best_set
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Branch 1: include best_v
        new_avail = available_mask & ~adj_bits[best_v] & ~(1 << best_v)
        bb_search(new_avail, current_size + 1, current_set | {best_v})
        
        # Branch 2: exclude best_v
        new_avail = available_mask & ~(1 << best_v)
        bb_search(new_avail, current_size, current_set)
    
    # Run B&B with limited nodes
    full_mask = (1 << num_valid) - 1
    if not time_check():
        max_nodes = 500000
        old_deadline = deadline
        deadline = min(deadline, time.time() + 25)  # cap B&B at 25s
        bb_search(full_mask, 0, set())
        deadline = old_deadline
    
    # Final local search
    best_list, best_size_final = local_search(list(best_set))
    
    # Verify and return
    result_vals = [valid_numbers[i] for i in best_list]
    result_set = set(result_vals)
    for a in result_set:
        if 2 * a in squares:
            break
        for b in result_set:
            if b >= a and a + b in squares:
                # Fallback to greedy
                fallback = greedy_min_degree()
                result_vals = [valid_numbers[i] for i in fallback]
                break
    
    return sorted(result_vals)
