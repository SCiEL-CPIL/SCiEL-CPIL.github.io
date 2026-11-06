import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 20
    deadline = time.time() + 110  # Safety margin
    
    # Heuristic fallback components (kept from original)
    def is_valid(A):
        K = A @ A
        np.fill_diagonal(K, 0)
        return np.max(K) <= 2

    def count_edges(A):
        return int(np.sum(A)) // 2

    def try_add_edge(A, u, v):
        if A[u, v] == 1:
            return False
        for w in range(n):
            if w == u or w == v:
                continue
            cn_uw = int(np.dot(A[u, :], A[:, w]))
            if A[v, w] == 1:
                if cn_uw >= 2:
                    return False
            cn_vw = int(np.dot(A[v, :], A[:, w]))
            if A[u, w] == 1:
                if cn_vw >= 2:
                    return False
        A[u, v] = 1
        A[v, u] = 1
        return True

    def initialize():
        A = np.zeros((n, n), dtype=int)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        return A

    def greedy_fill(A):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        if try_add_edge(A, u, v):
                            changed = True
        return A

    # --- MILP Formulation ---
    
    # 1. Generate pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pairs) # 190
    pair_index = {p: idx for idx, p in enumerate(pairs)}
    
    # Variables:
    # 0..num_pairs-1: x_ij
    # num_pairs..num_pairs + num_y - 1: y_ijk
    
    # For each pair (i,j), we have n-2 y variables corresponding to k != i,j
    # Let's structure y variables as y[p][k_idx] where p is pair index and k_idx is index in list of valid k's
    # Or just a flat list.
    
    y_vars = []
    y_index = {} # (p, k) -> global var index
    
    for p, (i, j) in enumerate(pairs):
        for k in range(n):
            if k != i and k != j:
                # Determine x variable indices for (i,k) and (j,k)
                # Note: x variables are only defined for i<k.
                if i < k:
                    x_ik_idx = pair_index[(i, k)]
                else:
                    x_ik_idx = pair_index[(k, i)]
                    
                if j < k:
                    x_jk_idx = pair_index[(j, k)]
                else:
                    x_jk_idx = pair_index[(k, j)]
                    
                y_idx = len(y_vars)
                y_vars.append((p, k, x_ik_idx, x_jk_idx))
                y_index[(p, k)] = y_idx
                
    num_y = len(y_vars)
    total_vars = num_pairs + num_y
    
    # Objective: maximize sum of x
    # scipy milp minimizes c @ x. So we minimize -sum(x)
    c = np.zeros(total_vars)
    for i in range(num_pairs):
        c[i] = -1.0
    
    # Bounds
    # x in [0,1], y in [0,1]
    # Integrality: 1 for all (since it's 0/1)
    integrality = np.ones(total_vars)
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    
    # Constraints
    # We need to build A_ub and b_ub or use LinearConstraint
    
    # 1. y_p_k <= x_ik
    # 2. y_p_k <= x_jk
    # 3. y_p_k >= x_ik + x_jk - 1  =>  -y_p_k + x_ik + x_jk <= 1
    # 4. sum_k y_p_k <= 2 for each p
    
    # Let's count constraints
    # For each y var: 3 constraints
    # For each pair p: 1 constraint
    num_constraints = 3 * num_y + num_pairs
    
    A_mat = np.zeros((num_constraints, total_vars))
    b_ub = np.zeros(num_constraints)
    
    constraint_idx = 0
    
    # Constraints for y variables
    for y_idx, (p, k, x_ik_idx, x_jk_idx) in enumerate(y_vars):
        global_y_idx = num_pairs + y_idx
        
        # y <= x_ik  => y - x_ik <= 0
        A_mat[constraint_idx, global_y_idx] = 1
        A_mat[constraint_idx, x_ik_idx] = -1
        b_ub[constraint_idx] = 0
        constraint_idx += 1
        
        # y <= x_jk  => y - x_jk <= 0
        A_mat[constraint_idx, global_y_idx] = 1
        A_mat[constraint_idx, x_jk_idx] = -1
        b_ub[constraint_idx] = 0
        constraint_idx += 1
        
        # -y + x_ik + x_jk <= 1
        A_mat[constraint_idx, global_y_idx] = -1
        A_mat[constraint_idx, x_ik_idx] = 1
        A_mat[constraint_idx, x_jk_idx] = 1
        b_ub[constraint_idx] = 1
        constraint_idx += 1
        
    # Constraints for sum of y for each pair
    for p in range(num_pairs):
        # sum_{k} y_{p,k} <= 2
        A_mat[constraint_idx, :] = 0 # Ensure row is zeroed (not strictly needed if initialized)
        for y_idx, (p_curr, k, _, _) in enumerate(y_vars):
            if p_curr == p:
                A_mat[constraint_idx, num_pairs + y_idx] = 1
        b_ub[constraint_idx] = 2
        constraint_idx += 1
        
    # Create LinearConstraint
    # A_mat x <= b_ub
    # And x >= 0 (handled by bounds)
    # Actually, LinearConstraint(A, lb, ub) means lb <= A x <= ub
    # We have A x <= b_ub. So lb = -inf, ub = b_ub.
    lin_cons = LinearConstraint(A_mat, -np.inf, b_ub)
    bounds = Bounds(lb, ub)
    
    # Run MILP
    best_A = None
    best_edges = 0
    
    # Fallback heuristic
    def run_heuristic():
        A = initialize()
        for _ in range(20):
            u, v = random.sample(range(n), 2)
            try_add_edge(A, u, v)
        A = greedy_fill(A)
        return A
        
    try:
        if time.time() < deadline - 5: # Leave time for conversion
            # Set a time limit for the solver itself
            # HiGHS time limit is in seconds
            res = milp(c, constraints=lin_cons, integrality=integrality, bounds=bounds, 
                       options={"time_limit": 100})
            
            if res.success and res.x is not None:
                # Extract solution
                A = np.zeros((n, n), dtype=int)
                for i in range(num_pairs):
                    u, v = pairs[i]
                    if res.x[i] > 0.5:
                        A[u, v] = 1
                        A[v, u] = 1
                
                if is_valid(A):
                    edges = count_edges(A)
                    if edges > best_edges:
                        best_edges = edges
                        best_A = A
                else:
                    print("MILP returned invalid solution")
                    
    except Exception as e:
        print(f"MILP failed: {e}")
        
    # If MILP didn't find a good solution or failed, use heuristic
    if best_A is None or best_edges < 30: # 30 is seed
        A = run_heuristic()
        if best_A is None:
            best_A = A
            best_edges = count_edges(A)
        else:
            # Maybe combine? Just pick better
            h_edges = count_edges(A)
            if h_edges > best_edges:
                best_A = A
                best_edges = h_edges
                
    # Convert to list
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v]:
                edges_list.append([u, v])
                
    return edges_list
