import time
import math
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    N = 256
    max_sum = 2 * N

    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)

    # 2. Filter vertices: 2*v must not be a square (diagonal constraint)
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)

    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1]

    # Map vertex value to index in valid_vertices
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}

    # 3. Build adjacency bitsets for the conflict graph
    # Edge (i, j) exists if valid_vertices[i] + valid_vertices[j] is a square
    adj_bits = [0] * n_vars
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)

    # 4. Dominance Reduction
    # A vertex u is dominated by v (v != u) if closed_neigh[v] is a subset of closed_neigh[u].
    closed_neigh = [adj_bits[i] | (1 << i) for i in range(n_vars)]

    # Sort by popcount of closed neighborhood (ascending)
    order = sorted(range(n_vars), key=lambda i: bin(closed_neigh[i]).count('1'))
    
    kept_indices = []
    for u_idx in order:
        is_dominated = False
        for v_idx in kept_indices:
            # Check if N[v] is subset of N[u]
            if (closed_neigh[v_idx] & closed_neigh[u_idx]) == closed_neigh[v_idx]:
                is_dominated = True
                break
        if not is_dominated:
            kept_indices.append(u_idx)

    n_reduced = len(kept_indices)
    if n_reduced == 0:
        return [valid_vertices[0]]

    # Map reduced index to original valid_vertices index
    reduced_to_orig = [0] * n_reduced
    for i, idx in enumerate(kept_indices):
        reduced_to_orig[i] = idx

    # Build constraints for the reduced graph
    # We need to find edges between kept vertices.
    constraint_pairs = []
    red_adj = [0] * n_reduced
    for i in range(n_reduced):
        orig_i = kept_indices[i]
        for j in range(i + 1, n_reduced):
            orig_j = kept_indices[j]
            if valid_vertices[orig_i] + valid_vertices[orig_j] in squares:
                constraint_pairs.append((i, j))
                red_adj[i] |= (1 << j)
                red_adj[j] |= (1 << i)

    # Helper to convert reduced solution to original values
    def get_final_set(reduced_sol):
        return [valid_vertices[reduced_to_orig[i]] for i in reduced_sol]

    # Helper to verify and return size
    def verify_and_size(sol_values):
        for v in sol_values:
            if 2 * v in squares:
                return 0
        n = len(sol_values)
        for i in range(n):
            for j in range(i + 1, n):
                if sol_values[i] + sol_values[j] in squares:
                    return 0
        return n

    # Greedy Solver for Initial Solution and Pruning Bounds
    def greedy_solve(free_mask, current_size=0):
        sol = []
        avail = free_mask
        while avail:
            # Pick vertex with min degree in remaining graph
            best_v = -1
            best_deg = float('inf')
            for v in range(n_reduced):
                if avail & (1 << v):
                    deg = bin(red_adj[v] & avail).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            sol.append(best_v)
            avail &= ~(red_adj[best_v] | (1 << v))
        return sol, current_size + len(sol)

    # Initial Greedy Solution
    initial_sol, best_size = greedy_solve((1 << n_reduced) - 1, 0)
    best_solution_set = set(initial_sol)
    
    # Branch and Bound
    # Node structure: (free_mask, fixed_ones_count, fixed_ones_set)
    # We use a stack for DFS
    stack = [( (1 << n_reduced) - 1, 0, set() )]
    
    # Precompute LP data for the full graph? No, subgraphs change.
    # We will build the LP matrix on the fly for the subgraph.
    
    def solve_lp(free_mask):
        """Solves LP relaxation for the subgraph defined by free_mask.
        Returns (upper_bound, solution_vector_dict)"""
        # Get list of free vertices
        free_verts = [i for i in range(n_reduced) if free_mask & (1 << i)]
        if not free_verts:
            return 0.0, {}
            
        n_local = len(free_verts)
        local_to_global = {i: free_verts[i] for i in range(n_local)}
        
        # Objective: Maximize sum x_i -> Minimize -sum x_i
        c = -np.ones(n_local)
        
        # Constraints: x_i + x_j <= 1 for all edges in subgraph
        # We need to find edges between free_verts
        A_rows = []
        b_rows = []
        
        for i_idx, i in enumerate(free_verts):
            # Neighbors of i that are also in free_verts
            neighbors_in_free = red_adj[i] & free_mask
            # Iterate through set bits of neighbors_in_free
            # To be efficient, we can just check all j in free_verts > i
            for j_idx in range(i_idx + 1, n_local):
                j = free_verts[j_idx]
                if red_adj[i] & (1 << j):
                    row = np.zeros(n_local)
                    row[i_idx] = 1.0
                    row[j_idx] = 1.0
                    A_rows.append(row)
                    b_rows.append(1.0)
                    
        if not A_rows:
            return float(n_local), {local_to_global[i]: 1.0 for i in range(n_local)}

        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        bounds = [(0, 1) for _ in range(n_local)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                # Maximize sum x -> res.fun is -sum x
                upper_bound = -res.fun
                sol_vec = res.x
                sol_dict = {local_to_global[i]: sol_vec[i] for i in range(n_local)}
                return upper_bound, sol_dict
            else:
                # If LP fails, return a loose bound
                return float(n_local), {}
        except Exception:
            return float(n_local), {}

    # Main B&B Loop
    while stack:
        if time.time() > deadline:
            break
            
        free_mask, fixed_count, fixed_set = stack.pop()
        
        # Lower Bound Pruning: If current fixed + optimistic greedy <= best, skip?
        # Actually, we use LP for upper bound.
        
        # Calculate Upper Bound
        # Quick bound: Greedy on free_mask
        _, greedy_bound_val = greedy_solve(free_mask, 0)
        if fixed_count + greedy_bound_val <= best_size:
            continue
            
        # LP Bound
        lp_bound, lp_sol = solve_lp(free_mask)
        
        if fixed_count + lp_bound <= best_size:
            continue
            
        # If LP solution is integral (or close to it), update best
        # Check if all x_i are close to 0 or 1
        is_integral = True
        candidate_set = set(fixed_set)
        for v, val in lp_sol.items():
            if val > 0.99:
                candidate_set.add(v)
            elif val < 0.01:
                pass
            else:
                is_integral = False
                break
        
        if is_integral:
            # Verify candidate
            # Since it's from LP relaxation, it might not be a valid independent set if there are numerical issues, 
            # but usually integral LP solutions for MIS are valid.
            # However, LP relaxation allows fractional, so if it's integral, it's a valid IS.
            if len(candidate_set) > best_size:
                # Verify
                valid = True
                for v in candidate_set:
                    # Check if v conflicts with any other in candidate_set
                    # We can just check if it's a valid IS
                    pass # Simplified: assume valid if integral
                # To be safe, verify
                # This is O(N^2), acceptable for small N
                is_valid = True
                cand_list = list(candidate_set)
                for i in range(len(cand_list)):
                    for j in range(i+1, len(cand_list)):
                        if red_adj[cand_list[i]] & (1 << cand_list[j]):
                            is_valid = False
                            break
                    if not is_valid: break
                if is_valid:
                    best_size = len(candidate_set)
                    best_solution_set = candidate_set
            continue

        # Branching
        # Find variable to branch on
        # Choose fractional variable with highest degree in free subgraph
        branch_var = -1
        max_deg = -1
        for v, val in lp_sol.items():
            if 0.1 < val < 0.9:
                deg = bin(red_adj[v] & free_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    branch_var = v
                    
        if branch_var == -1:
            # No clear fractional var? Pick first
            for v, val in lp_sol.items():
                if 0.1 < val < 0.9:
                    branch_var = v
                    break
            if branch_var == -1:
                continue

        # Branch 1: Include branch_var
        new_free_mask_1 = free_mask & ~(red_adj[branch_var] | (1 << branch_var))
        new_fixed_set_1 = set(fixed_set)
        new_fixed_set_1.add(branch_var)
        stack.append((new_free_mask_1, fixed_count + 1, new_fixed_set_1))
        
        # Branch 2: Exclude branch_var
        new_free_mask_2 = free_mask & ~(1 << branch_var)
        stack.append((new_free_mask_2, fixed_count, set(fixed_set)))

    # Final Polish: Try to add any remaining vertex to best_solution_set
    current_set = set(best_solution_set)
    current_mask = 0
    for i in current_set:
        current_mask |= (1 << i)
        
    improved = True
    while improved and (time.time() - start_time) < deadline - 2.0:
        improved = False
        for v in range(n_reduced):
            if v not in current_set:
                if (red_adj[v] & current_mask) == 0:
                    current_set.add(v)
                    current_mask |= (1 << v)
                    improved = True
                    break
                    
    final_set = get_final_set(sorted(current_set))
    
    # Final Verification
    size = verify_and_size(final_set)
    if size == 0:
        return [valid_vertices[0]]
        
    return final_set
