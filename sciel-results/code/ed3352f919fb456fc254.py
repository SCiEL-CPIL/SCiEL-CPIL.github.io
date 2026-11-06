import random
import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 18
    # Precompute indices
    edge_indices = []
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[(u, v)] = idx
            edge_indices.append((u, v))
            idx += 1
    num_edges = len(edge_indices)
    
    tri_indices = []
    tri_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                tri_map[(u, v, w)] = idx
                tri_indices.append((u, v, w))
                idx += 1
    num_tris = len(tri_indices)
    
    # Total variables: num_edges + num_tris
    # x[0:num_edges] are e_uv
    # x[num_edges:num_edges+num_tris] are t_uvw
    
    total_vars = num_edges + num_tris
    
    # Helper to get triangle indices in a set S
    # S is a frozenset of vertices
    # We need to precompute or generate on the fly
    # Since we iterate m, and for each m we check all subsets of size m+1,
    # it's better to precompute the list of subsets for each size.
    
    # However, generating all subsets of size 6 for n=18 is 18564.
    # For each subset, we need the list of triangles within it.
    # Triangles in S: triples (u,v,w) in S.
    
    # Let's try m=5 first (size 6 subsets)
    # If that fails, try m=6 (size 7 subsets)
    
    # We will iterate m from 5 upwards.
    
    # Precompute subsets for sizes 6, 7, 8 just in case
    # It might be heavy to store all, but 18k is small.
    
    from itertools import combinations
    
    # Constraints for K4-free
    # For each 4-set Q, sum(e_e for e in Q) <= 5
    # e_e corresponds to x[edge_map[e]]
    
    A_k4_rows = []
    b_k4 = []
    
    for Q in combinations(range(n), 4):
        row = np.zeros(num_edges)
        count = 0
        for u, v in combinations(Q, 2):
            e_idx = edge_map[(u, v)]
            row[e_idx] = 1.0
            count += 1
        A_k4_rows.append(row)
        b_k4.append(5.0) # sum <= 5
        
    A_k4 = np.array(A_k4_rows) if A_k4_rows else np.empty((0, num_edges))
    
    # Constraints for Triangle Logic
    # t <= e_uv, t <= e_uw, t <= e_vw
    # t >= e_uv + e_uw + e_vw - 2
    
    # We can combine these into linear constraints.
    # Let's build the full A matrix for the ILP.
    # We need to add constraints for the specific m.
    
    # We'll write a helper to solve for a given m
    def solve_for_m(m):
        # Target: All sets of size m+1 must contain a triangle.
        # Constraint: For all S with |S| = m+1, sum(t_tri for tri in S) >= 1
        
        # Build constraints for S
        A_set_rows = []
        b_set = []
        
        # Generate subsets of size m+1
        target_size = m + 1
        if target_size > n:
            return None
            
        for S in combinations(range(n), target_size):
            S_set = set(S)
            row = np.zeros(total_vars)
            # Find all triangles in S
            for u, v, w in combinations(S, 3):
                t_idx = num_edges + tri_map[(u, v, w)]
                row[t_idx] = 1.0
            A_set_rows.append(row)
            b_set.append(1.0)
            
        A_set = np.array(A_set_rows) if A_set_rows else np.empty((0, total_vars))
        b_set_arr = np.array(b_set)
        
        # Combine with K4 and Triangle Logic constraints
        # Triangle Logic Constraints:
        # 1. t - e_uv <= 0
        # 2. t - e_uw <= 0
        # 3. t - e_vw <= 0
        # 4. t - e_uv - e_uw - e_vw >= -2  =>  -t + e_uv + e_uw + e_vw <= 2
        
        A_tri_rows = []
        b_tri = []
        
        for i, (u, v, w) in enumerate(tri_indices):
            t_idx = num_edges + i
            e_uv_idx = edge_map[(u, v)]
            e_uw_idx = edge_map[(u, w)]
            e_vw_idx = edge_map[(v, w)]
            
            # t - e_uv <= 0
            r1 = np.zeros(total_vars)
            r1[t_idx] = 1.0
            r1[e_uv_idx] = -1.0
            A_tri_rows.append(r1)
            b_tri.append(0.0)
            
            # t - e_uw <= 0
            r2 = np.zeros(total_vars)
            r2[t_idx] = 1.0
            r2[e_uw_idx] = -1.0
            A_tri_rows.append(r2)
            b_tri.append(0.0)
            
            # t - e_vw <= 0
            r3 = np.zeros(total_vars)
            r3[t_idx] = 1.0
            r3[e_vw_idx] = -1.0
            A_tri_rows.append(r3)
            b_tri.append(0.0)
            
            # -t + e_uv + e_uw + e_vw <= 2
            r4 = np.zeros(total_vars)
            r4[t_idx] = -1.0
            r4[e_uv_idx] = 1.0
            r4[e_uw_idx] = 1.0
            r4[e_vw_idx] = 1.0
            A_tri_rows.append(r4)
            b_tri.append(2.0)
            
        A_tri = np.array(A_tri_rows) if A_tri_rows else np.empty((0, total_vars))
        b_tri_arr = np.array(b_tri)
        
        # Combine all A and b
        # Constraints:
        # A_k4 x <= b_k4
        # A_tri x <= b_tri
        # A_set x >= b_set  =>  -A_set x <= -b_set
        
        A_all = np.vstack([A_k4, A_tri, -A_set])
        b_all = np.concatenate([b_k4, b_tri_arr, -b_set_arr])
        
        # Objective: Minimize 0 (Feasibility problem)
        # Or minimize sum of edges? No, just feasibility.
        # scipy.milp requires c. c = 0.
        c = np.zeros(total_vars)
        
        # Bounds: 0 <= x <= 1
        integrality = np.ones(total_vars)
        bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
        
        constraint = LinearConstraint(A_all, ub=b_all)
        
        try:
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            if res.status == 0:
                # Extract edges
                x = res.x
                edges = []
                for i, (u, v) in enumerate(edge_indices):
                    if x[i] > 0.5:
                        edges.append([u, v])
                return edges
            else:
                return None
        except Exception as e:
            return None

    # Try m from 5 to 10
    for m in range(5, 11):
        edges = solve_for_m(m)
        if edges is not None:
            return edges
            
    # Fallback: Turan T(18,3)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.append([u, v])
    return edges
