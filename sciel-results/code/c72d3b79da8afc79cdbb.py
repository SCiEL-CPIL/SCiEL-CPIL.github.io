"""Matrix multiplication 2,2,10 via block-diagonal Strassen (rank 35)."""
import time
import sys

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5 blocks of 2x2
    
    # Rank is 5 * 7 = 35
    rank = 35
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 35)
    # V: (m*p, rank) = (20, 35)
    # W: (p*n, rank) = (20, 35)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for a single 2x2 * 2x2 block
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Define the 7 Strassen terms for one block
    # Each term is a list of (u_idx, u_val, v_local_idx, v_val, w_local_idx, w_val)
    # We will map local v and w indices to global indices inside the loop
    
    # Term 1: M1 = (a+d)(e+h)
    # U: a(1), d(1) -> U[0]+=1, U[3]+=1
    # V: e(1), h(1) -> V_loc[0]+=1, V_loc[3]+=1
    # W: C00(-1), C11(1) -> Wait, standard Strassen:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's list contributions per term:
    
    # M1: (a+d)(e+h)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 0, 0, 1] (Contributes +1 to C00 and +1 to C11? No, check formula)
    # Formula: C00 = M1 - M4 + M5 - M7. So M1 contributes +1 to C00.
    # Formula: C11 = M1 - M2 + M3 + M6. So M1 contributes +1 to C11.
    # So W_local: [1, 0, 0, 1]
    
    # M2: (c+d)e
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: C10(+1), C11(-1). So W_local: [0, 0, 1, -1]
    
    # M3: a(f-h)
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1]
    # W: C01(+1), C11(+1). So W_local: [0, 1, 0, 1]
    
    # M4: d(g-e)
    # U: [0, 0, 0, 1]
    # V: [0, 0, 1, 0] for g? No, V is [e, f, g, h].
    # V: g is index 2, e is index 0.
    # V: [0, 0, 1, 0] for g? No, V[2] is g. V[0] is e.
    # V: [0, 0, 1, 0] is wrong. V should be [0, 0, 1, 0] if only g?
    # M4 = d(g-e).
    # V: e(-1), g(1). So V_local: [0, 0, 1, 0] is wrong. V[0] is e, V[2] is g.
    # V_local: [0, 0, 1, 0] -> V[2]=1, V[0]=-1. So [ -1, 0, 1, 0 ]? 
    # Let's stick to indices: 0:e, 1:f, 2:g, 3:h.
    # M4: d(g-e). V: e=-1, g=1. So V_local: [-1, 0, 1, 0].
    # W: C00(-1), C10(+1). So W_local: [-1, 0, 1, 0].
    
    # M5: (a+b)h
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # W: C00(+1), C01(+1). So W_local: [1, 1, 0, 0].
    
    # M6: (c-a)(e+f)
    # U: [0, 0, 1, 0] for c? U[2]=1. U[0]=-1 for a. So [-1, 0, 1, 0].
    # V: [1, 1, 0, 0]
    # W: C11(+1). So W_local: [0, 0, 0, 1].
    
    # M7: (b-d)(g+h)
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # W: C00(-1). So W_local: [-1, 0, 0, 0].
    
    # Summary of terms (U, V_loc, W_loc):
    terms = [
        # (U_vals, V_vals, W_vals)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),    # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 0]),  # M7
    ]
    
    for t in range(num_blocks):
        col_base = t * 7
        
        # Global index mappings for V and W in this block
        # V global indices for local 0,1,2,3:
        # local 0 (j=0, k=2t) -> 0*10 + 2t = 2t
        # local 1 (j=0, k=2t+1) -> 0*10 + 2t+1 = 2t+1
        # local 2 (j=1, k=2t) -> 1*10 + 2t = 10+2t
        # local 3 (j=1, k=2t+1) -> 1*10 + 2t+1 = 11+2t
        v_indices = [2*t, 2*t+1, 10+2*t, 11+2*t]
        
        # W global indices for local 0,1,2,3:
        # local 0 (k=2t, i=0) -> 2t*2 + 0 = 4t
        # local 1 (k=2t, i=1) -> 2t*2 + 1 = 4t+1
        # local 2 (k=2t+1, i=0) -> (2t+1)*2 + 0 = 4t+2
        # local 3 (k=2t+1, i=1) -> (2t+1)*2 + 1 = 4t+3
        w_indices = [4*t, 4*t+1, 4*t+2, 4*t+3]
        
        for c in range(7):
            col = col_base + c
            u_vals, v_vals, w_vals = terms[c]
            
            # Fill U
            for i in range(4):
                if u_vals[i] != 0:
                    U[i][col] = u_vals[i]
            
            # Fill V
            for i in range(4):
                if v_vals[i] != 0:
                    V[v_indices[i]][col] = v_vals[i]
            
            # Fill W
            for i in range(4):
                if w_vals[i] != 0:
                    W[w_indices[i]][col] = w_vals[i]
    
    # Verification
    if time.time() < deadline:
        try:
            import numpy as np
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            rng = np.random.default_rng(42)
            A_test = rng.standard_normal((n, m))
            B_test = rng.standard_normal((m, p))
            C_expected = A_test @ B_test
            C_computed = np.zeros((n, p))
            
            # Efficient check:
            # C[i,k] = sum_j sum_c A[i,j] U[i,j,c] V[j,k,c] W[k,i,c]
            # This is effectively sum_c (U_c^T A) (V_c^T B) W_c
            # But let's just do the triple sum for small matrices
            
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for j in range(m):
                        for c in range(rank):
                            val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    C_computed[i, k] = val
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                # Fallback to schoolbook if Strassen implementation has a bug
                # Rank 40
                rank_fb = n * m * p
                U_fb = [[0] * rank_fb for _ in range(n * m)]
                V_fb = [[0] * rank_fb for _ in range(m * p)]
                W_fb = [[0] * rank_fb for _ in range(p * n)]
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i*m+j][idx] = 1.0
                            V_fb[j*p+k][idx] = 1.0
                            W_fb[k*n+i][idx] = 1.0
                            idx += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
