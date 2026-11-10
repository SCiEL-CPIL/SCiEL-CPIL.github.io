"""Rewritten program: branch-and-bound on conflict graph with LP relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from itertools import combinations
import math

def construct():
    n = 256
    # Build conflict graph: edge (i,j) if i+j is a perfect square (for i<=j, including i=j)
    # Vertices are 1..256, indexed 0..255
    
    # Precompute perfect squares up to 512
    max_sum = 2 * 256
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # Build adjacency: adj[i] is set of j > i such that (i+1)+(j+1) is a square
    # Also self-loops: if 2*(i+1) is a square, vertex i cannot be selected
    adj = [set() for _ in range(n)]
    self_conflict = [False] * n
    
    for i in range(n):
        val_i = i + 1
        # Check self-conflict
        if 2 * val_i in squares:
            self_conflict[i] = True
        for j in range(i + 1, n):
            val_j = j + 1
            if val_i + val_j in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    # Filter out vertices with self-conflict (they can never be in the set)
    # Actually, if 2*a is a perfect square, a cannot be in the set.
    # So we remove those vertices entirely.
    
    valid_indices = [i for i in range(n) if not self_conflict[i]]
    num_valid = len(valid_indices)
    
    if num_valid == 0:
        return [1]  # fallback, though this shouldn't happen
    
    # Remap to 0..num_valid-1
    index_map = {orig: new for new, orig in enumerate(valid_indices)}
    
    # Build new adjacency in compact form
    compact_adj = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        orig_i = valid_indices[i]
        for j in adj[orig_i]:
            if j in index_map:
                compact_adj[i].add(index_map[j])
    
    # Degrees for heuristics
    degrees = [len(compact_adj[i]) for i in range(num_valid)]
    
    # Branch and bound
    best_set = set()
    best_size = 0
    
    # Sort vertices by degree descending for branching
    vertex_order = sorted(range(num_valid), key=lambda x: -degrees[x])
    
    # For LP relaxation, we need the constraint matrix
    # Variables x_i in [0,1], maximize sum x_i
    # Constraints: for each edge (i,j), x_i + x_j <= 1
    # Also 0 <= x_i <= 1
    
    edges = []
    for i in range(num_valid):
        for j in compact_adj[i]:
            if i < j:
                edges.append((i, j))
    
    num_edges = len(edges)
    
    # Build A_ub and b_ub for linprog
    # Each edge gives one row: x_i + x_j <= 1
    A_ub = np.zeros((num_edges, num_valid))
    for idx, (i, j) in enumerate(edges):
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
    b_ub = np.ones(num_edges)
    
    c = np.ones(num_valid)  # maximize sum x_i, linprog minimizes, so use -c
    
    # Initial LP solve for upper bound
    def solve_lp(active_vars):
        """Solve LP relaxation on active variables (those not yet branched)."""
        if not active_vars:
            return 0.0
        n_active = len(active_vars)
        if n_active == 0:
            return 0.0
        
        # Build sub-matrix for active variables
        A_sub = A_ub[:, active_vars]
        c_sub = c[active_vars]
        
        # Only include edges that have both endpoints in active_vars
        edge_mask = []
        for idx, (i, j) in enumerate(edges):
            if i in active_vars_set and j in active_vars_set:
                edge_mask.append(idx)
        
        if not edge_mask:
            return float(n_active)
        
        A_sub = A_ub[edge_mask, :][:, active_vars]
        
        res = linprog(-c_sub, A_ub=A_sub, b_ub=np.ones(len(edge_mask)),
                      bounds=[(0, 1)] * n_active, method='highs')
        if res.success:
            return -res.fun
        else:
            return float(n_active)  # fallback
    
    active_vars_set = set(range(num_valid))
    
    # Global upper bound from initial LP
    initial_ub = solve_lp(list(range(num_valid)))
    
    # Branch and bound
    # State: (active_set, current_selected, depth)
    # We branch on the vertex with highest degree in active set
    
    # Use a stack-based approach
    # Each node: (active_vars_list, selected_set, ub)
    
    import heapq
    
    # We'll use a simple DFS with pruning
    
    def branch_and_bound(active_vars, selected, current_ub):
        nonlocal best_set, best_size
        
        if not active_vars:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = set(selected)
            return
        
        # Upper bound: current selected + LP bound on active
        if current_ub + len(selected) <= best_size:
            return
        
        # Try to find a greedy solution to update best
        # Greedy: pick vertex with min degree in active, add to selected, remove its neighbors
        greedy_selected = set(selected)
        greedy_active = list(active_vars)
        greedy_adj = {v: set(compact_adj[v]) & set(greedy_active) for v in greedy_active}
        
        while greedy_active:
            # Pick vertex with minimum degree
            min_deg_vertex = min(greedy_active, key=lambda v: len(greedy_adj[v]))
            greedy_selected.add(min_deg_vertex)
            neighbors = greedy_adj[min_deg_vertex]
            greedy_active = [v for v in greedy_active if v != min_deg_vertex and v not in neighbors]
            for v in neighbors:
                if v in greedy_adj:
                    del greedy_adj[v]
            # Update degrees for remaining
            for v in greedy_active:
                if min_deg_vertex in greedy_adj[v]:
                    greedy_adj[v].discard(min_deg_vertex)
        
        if len(greedy_selected) > best_size:
            best_size = len(greedy_selected)
            best_set = set(greedy_selected)
        
        if current_ub + len(selected) <= best_size:
            return
        
        # Branch on vertex with highest degree in active
        v = max(active_vars, key=lambda x: len(compact_adj[x]) & set(active_vars))
        
        # Branch 1: include v
        new_selected = list(selected) + [v]
        neighbors_of_v = compact_adj[v]
        new_active = [u for u in active_vars if u != v and u not in neighbors_of_v]
        
        # Compute LP bound for new_active
        if new_active:
            new_active_set = set(new_active)
            # Quick bound: just count
            ub1 = len(new_active)  # simple bound
            # Could do LP but it's expensive; use simple bound for now
            branch_and_bound(new_active, new_selected, ub1)
        
        # Branch 2: exclude v
        new_active2 = [u for u in active_vars if u != v]
        if new_active2:
            ub2 = len(new_active2)
            branch_and_bound(new_active2, selected, ub2)
    
    # Start branch and bound
    branch_and_bound(list(range(num_valid)), [], initial_ub)
    
    # Convert back to original values
    if best_set:
        result = sorted([valid_indices[i] + 1 for i in best_set])
    else:
        # Fallback: just pick some non-conflicting vertices
        result = []
        used = set()
        for i in valid_indices:
            if i not in used:
                result.append(i + 1)
                used.add(i)
                used.update(adj[i])
    
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
