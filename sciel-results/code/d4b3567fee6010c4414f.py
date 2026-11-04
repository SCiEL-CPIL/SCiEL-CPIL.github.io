"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define 4 disjoint 2x2x2 blocks
    # Box 1: I={0,1}, J={0,1}, K={0,1}
    # Box 2: I={0,1}, J={2,3}, K={0,1}
    # Box 3: I={1,2}, J={0,1}, K={2,3}
    # Box 4: I={0,1}, J={2,3}, K={3,4}
    
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [0, 1]),
        ([1, 2], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [3, 4])
    ]
    
    # Map global indices to local indices for each block
    # We will generate the U, V, W vectors for each block
    
    U = []
    V = []
    W = []
    
    # Strassen coefficients for the 7 multiplications
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # u coefficients for A (local indices 0,1 for i, 0,1 for j)
    # v coefficients for B (local indices 0,1 for j, 0,1 for k)
    # w coefficients for C (local indices 0,1 for k, 0,1 for i)
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u is size 4 (2x2), v is size 4 (2x2), w is size 4 (2x2)
        # Indexing: u[i*2+j], v[j*2+k], w[k*2+i]
        
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]), # C00+=M1, C11-=M1? 
        # Wait, let's use the explicit reconstruction formulas.
        # C00 = M1 - M4 - M5 + M7
        # C01 = M1 + M3 - M5 - M6
        # C10 = M2 + M4 - M1 + M7
        # C11 = M2 + M3 + M5 - M6
        
        # So w for M1 is [1, 1, -1, 0] ?
        # w[k*2+i]
        # C00 (k=0,i=0) -> w[0] += 1
        # C01 (k=1,i=0) -> w[2] += 1
        # C10 (k=0,i=1) -> w[1] += -1
        # C11 (k=1,i=1) -> w[3] += 0
        # So w_M1 = [1, -1, 1, 0]
        
        # Let's re-verify Strassen formulas.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 - M5 + M7
        # C01 = M1 + M3 - M5 - M6
        # C10 = M2 + M4 - M1 + M7
        # C11 = M2 + M3 + M5 - M6
        
        # w_M1: C00(+1), C01(+1), C10(-1), C11(0) -> [1, -1, 1, 0]
        # w_M2: C00(0), C01(0), C10(+1), C11(+1) -> [0, 1, 0, 1]
        # w_M3: C00(0), C01(+1), C10(0), C11(+1) -> [0, 0, 1, 1]
        # w_M4: C00(-1), C01(0), C10(+1), C11(0) -> [-1, 1, 0, 0]
        # w_M5: C00(-1), C01(-1), C10(0), C11(+1) -> [-1, 0, -1, 1]
        # w_M6: C00(0), C01(-1), C10(0), C11(-1) -> [0, 0, -1, -1]
        # w_M7: C00(+1), C01(0), C10(+1), C11(0) -> [1, 1, 0, 0]
        
        # u_M1: A00(1), A11(1) -> [1, 0, 0, 1]
        # v_M1: B00(1), B11(1) -> [1, 0, 0, 1]
        
        # u_M2: A10(1), A11(1) -> [0, 0, 1, 1]
        # v_M2: B00(1) -> [1, 0, 0, 0]
        
        # u_M3: A00(1) -> [1, 0, 0, 0]
        # v_M3: B10(1), B11(-1) -> [0, 0, 1, -1]
        
        # u_M4: A11(1) -> [0, 0, 0, 1]
        # v_M4: B01(1), B00(-1) -> [-1, 1, 0, 0]
        
        # u_M5: A00(1), A01(1) -> [1, 1, 0, 0]
        # v_M5: B11(1) -> [0, 0, 0, 1]
        
        # u_M6: A10(1), A00(-1) -> [-1, 0, 1, 0]
        # v_M6: B00(1), B01(1) -> [1, 1, 0, 0]
        
        # u_M7: A01(1), A11(-1) -> [0, 1, 0, -1]
        # v_M7: B10(1), B11(1) -> [0, 0, 1, 1]
    ]
    
    strassen_data = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, -1, 1, 0]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 1, 0, 0], [-1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, -1, 1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, -1, -1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0])
    ]
    
    # Initialize full size vectors
    rank = 0
    U_full = []
    V_full = []
    W_full = []
    
    # Track which triples are covered by Strassen blocks
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
    
    # Add Strassen terms
    for I, J, K in blocks:
        for u_loc, v_loc, w_loc in strassen_data:
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            # Map local u (i', j') to global (i, j)
            for i_prime in range(2):
                for j_prime in range(2):
                    val = u_loc[i_prime * 2 + j_prime]
                    if val != 0:
                        i_glob = I[i_prime]
                        j_glob = J[j_prime]
                        u_vec[i_glob * m + j_glob] += val
            
            # Map local v (j', k') to global (j, k)
            for j_prime in range(2):
                for k_prime in range(2):
                    val = v_loc[j_prime * 2 + k_prime]
                    if val != 0:
                        j_glob = J[j_prime]
                        k_glob = K[k_prime]
                        v_vec[j_glob * p + k_glob] += val
            
            # Map local w (k', i') to global (k, i)
            for k_prime in range(2):
                for i_prime in range(2):
                    val = w_loc[k_prime * 2 + i_prime]
                    if val != 0:
                        k_glob = K[k_prime]
                        i_glob = I[i_prime]
                        w_vec[k_glob * n + i_glob] += val
            
            U_full.append(u_vec)
            V_full.append(v_vec)
            W_full.append(w_vec)
            rank += 1
    
    # Add Schoolbook terms for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u_vec = [0] * (n * m)
                    v_vec = [0] * (m * p)
                    w_vec = [0] * (p * n)
                    
                    u_vec[i * m + j] = 1
                    v_vec[j * p + k] = 1
                    w_vec[k * n + i] = 1
                    
                    U_full.append(u_vec)
                    V_full.append(v_vec)
                    W_full.append(w_vec)
                    rank += 1
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
