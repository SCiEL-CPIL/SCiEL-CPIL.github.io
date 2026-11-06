import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    N = 256

    # 1. Precompute ALL APs
    ap_indices = []
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
            
            # Each prefix is a valid AP
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_indices.append(idx_list)
    
    num_aps = len(ap_indices)

    # 2. Formulate MILP
    # Variables: v_0, ..., v_{N-1} (binary), D (continuous >= 0)
    # Minimize D
    # Constraints for each AP A:
    # Let S_A = sum_{i in A} x_i where x_i = 2v_i - 1
    # -D <= S_A <= D
    # S_A = 2*sum(v_i) - |A|
    # 
    # Upper bound: 2*sum(v_i) - |A| <= D  =>  2*sum(v_i) - D <= |A|
    # Lower bound: -(2*sum(v_i) - |A|) <= D => -2*sum(v_i) - D <= -|A|
    
    n_vars = N + 1
    d_idx = N  # Index of D in the variable vector
    
    # Objective: minimize D
    c = np.zeros(n_vars)
    c[d_idx] = 1.0
    
    # Build sparse constraint matrix
    num_constraints = 2 * num_aps
    
    rows = []
    cols = []
    data = []
    b_ub = []
    
    row_idx = 0
    for ap in ap_indices:
        L = len(ap)
        
        # Constraint 1: 2*sum(v_i) - D <= L
        for i in ap:
            rows.append(row_idx)
            cols.append(i)
            data.append(2.0)
        rows.append(row_idx)
        cols.append(d_idx)
        data.append(-1.0)
        b_ub.append(L)
        
        # Constraint 2: -2*sum(v_i) - D <= -L
        for i in ap:
            rows.append(row_idx + 1)
            cols.append(i)
            data.append(-2.0)
        rows.append(row_idx + 1)
        cols.append(d_idx)
        data.append(-1.0)
        b_ub.append(-L)
        
        row_idx += 2

    A_ub = sp.csr_matrix((data, (rows, cols)), shape=(num_constraints, n_vars))
    b_ub = np.array(b_ub)
    
    # Bounds: v_i in [0, 1], D in [0, inf)
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[d_idx] = np.inf
    
    # Integrality: v_i are integer (binary), D is continuous
    integrality = np.zeros(n_vars)
    integrality[:N] = 1
    
    try:
        # Use HiGHS via scipy.optimize.milp
        # time_limit is in seconds
        remaining_time = time_limit - (time.time() - start_time) - 2.0
        if remaining_time > 1.0:
            res = milp(c=c,
                       constraints=LinearConstraint(A_ub, -np.inf, b_ub),
                       integrality=integrality,
                       bounds=Bounds(lb, ub),
                       options={"time_limit": remaining_time, "mip_rel_gap": 0.0})
            
            if res.success and res.status == 0:
                v = res.x[:N]
                # Convert binary to signs -1, 1
                # v is in [0, 1]. 0 -> -1, 1 -> 1
                signs = np.where(v > 0.5, 1, -1).astype(int)
                
                # Verify
                max_dis = 0
                for ap in ap_indices:
                    s = np.sum(signs[ap])
                    if abs(s) > max_dis:
                        max_dis = abs(s)
                
                return [int(x) for x in signs]
            else:
                # Fallback to Thue-Morse if MILP fails or times out
                pass
                
    except Exception:
        # Fallback to Thue-Morse if MILP fails
        pass

    # Fallback: Thue-Morse
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    tm = thue_morse(N)
    return [int(x) for x in tm]
