import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 6
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    # Use schoolbook rank-48 decomposition as fallback, then try Strassen
    # First, let's build the correct Strassen-based decomposition
    
    rank = 42
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen terms for 2x2x2: (a_vec, b_vec, c_vec)
    # Local indices: a[li*2+lj], b[lj*2+lk], c[lk*2+li]
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    current_rank = 0
    
    # 6 blocks: (i_off, j_off, k_off)
    # i_off: row offset for A and C (0 or 0, since n=2)
    # j_off: col offset for A and row offset for B (0 or 2)
    # k_off: col offset for B and col offset for C (0, 2, or 4)
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
    ]
    
    for i_off, j_off, k_off in blocks:
        for a_vec, b_vec, c_vec in strassen_terms:
            # Map local A indices to global
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    val = a_vec[local_idx]
                    if val != 0:
                        global_a = li * m + (lj + j_off)
                        U[global_a, current_rank] += val
            
            # Map local B indices to global
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    val = b_vec[local_idx]
                    if val != 0:
                        global_b = (lj + j_off) * p + (lk + k_off)
                        V[global_b, current_rank] += val
            
            # Map local C indices to global
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    val = c_vec[local_idx]
                    if val != 0:
                        global_c = (lk + k_off) * n + li
                        W[global_c, current_rank] += val
            
            current_rank += 1
            
            if time.time() > deadline:
                break
    
    # Verify the decomposition
    # T[a,b,c] should equal sum_r U[a,r] * V[b,r] * W[c,r]
    # For matrix multiplication, T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # else 0
    
    # Check a few coordinates to debug
    def check_tensor():
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Compute expected value
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    
                    # Compute actual value
                    actual = 0.0
                    for r in range(rank):
                        actual += U[a, r] * V[b, r] * W[c, r]
                    
                    if abs(expected - actual) > 1e-10:
                        return False, (a, b, c), expected, actual
        return True, None, None, None
    
    valid, coord, exp, act = check_tensor()
    
    if not valid:
        # Fallback to schoolbook rank-48
        rank = 48
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, r] = 1.0
                    V[b, r] = 1.0
                    W[c, r] = 1.0
                    r += 1
        
        U_out = U.tolist()
        V_out = V.tolist()
        W_out = W.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
