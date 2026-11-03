import numpy as np
from scipy.optimize import linprog

N = 100

def construct():
    # We want to find h that maximizes min_k sum |h_i + h_{i-k} - 1|
    # Let T be the lower bound for this sum.
    # Maximize T.
    
    # Variables:
    # h: N vars
    # z: N * (N-1) vars
    # Total: N + N*(N-1)
    
    n_h = N
    n_z = N * (N - 1)
    n_vars = n_h + n_z
    
    # Objective: Maximize T -> Minimize -T
    # But T is not a variable in the standard form unless we add it.
    # Let's add T as a variable.
    # Index: 0..N-1 for h, N..N+N*(N-1)-1 for z, Last for T.
    
    idx_T = n_vars
    n_vars += 1
    
    c = np.zeros(n_vars)
    c[idx_T] = -1 # Minimize -T
    
    A_ub = []
    b_ub = []
    
    # Constraints for z >= h_i + h_j - 1
    # z - h_i - h_j >= -1  =>  -z + h_i + h_j <= 1
    # And z >= 1 - h_i - h_j => z + h_i + h_j >= 1 => -z - h_i - h_j <= -1
    
    # We will construct A_ub and b_ub for linprog (minimize c^T x s.t. A_ub x <= b_ub, A_eq x == b_eq, bounds)
    
    # 1. z_{i,k} - h_i - h_{i-k} >= -1  =>  -z_{i,k} + h_i + h_{i-k} <= 1
    # 2. z_{i,k} + h_i + h_{i-k} >= 1   =>  -z_{i,k} - h_i - h_{i-k} <= -1
    
    # 3. sum z_{i,k} >= T  =>  -sum z_{i,k} + T <= 0
    
    # 4. sum h_i = N/2 (Equality)
    
    # 5. Bounds on h: 0 <= h <= 1
    # 6. Bounds on z: z >= 0
    
    # Let's build the matrices.
    # This might be memory heavy if done naively, but N=100 is small.
    
    rows = []
    bounds = [(0, 1)] * n_h + [(0, None)] * n_z + [(0, None)] # T >= 0
    
    for k in range(1, N):
        for i in range(N):
            j = (i - k) % N
            idx_z = n_h + k * N + i
            
            # Constraint 1: -z + h_i + h_j <= 1
            row1 = np.zeros(n_vars)
            row1[idx_z] = -1
            row1[i] = 1
            row1[j] = 1
            rows.append(row1)
            b_ub.append(1)
            
            # Constraint 2: -z - h_i - h_j <= -1
            row2 = np.zeros(n_vars)
            row2[idx_z] = -1
            row2[i] = -1
            row2[j] = -1
            rows.append(row2)
            b_ub.append(-1)
            
        # Constraint 3: -sum z_{i,k} + T <= 0
        row3 = np.zeros(n_vars)
        row3[idx_T] = 1
        for i in range(N):
            idx_z = n_h + k * N + i
            row3[idx_z] = -1
        rows.append(row3)
        b_ub.append(0)
        
    A_ub = np.array(rows)
    b_ub = np.array(b_ub)
    
    # Equality constraints
    A_eq = []
    b_eq = []
    
    # Sum h = N/2
    row_eq = np.zeros(n_vars)
    for i in range(N):
        row_eq[i] = 1
    A_eq.append(row_eq)
    b_eq.append(N / 2.0)
    
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        h = res.x[:N]
        # Clean up numerical errors
        h = np.clip(h, 0, 1)
        # Ensure sum is exactly N/2?
        # The LP should satisfy it.
        return h
    else:
        # Fallback to uniform
        return np.full(N, 0.5)
