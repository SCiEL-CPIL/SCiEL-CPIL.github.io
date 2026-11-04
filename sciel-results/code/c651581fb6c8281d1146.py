"""Rewritten program using Strassen-like block decomposition for lower rank."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists (or arrays) and then convert to list.
    # Rank is estimated at 55.
    R = 55
    
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    rank = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal rank
        if rank >= R:
            return
        U[u_idx, rank] = u_val
        V[v_idx, rank] = v_val
        W[w_idx, rank] = w_val
        rank += 1

    # Strassen decomposition for 2x2x2 block
    # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    def add_strassen_block(i0, j0, k0):
        # Indices for U: i*m + j
        # Indices for V: j*p + k
        # Indices for W: k*n + i
        
        # Term 1: (A00+A11)(B00+B11)
        # u: i0,j0:1; i0+1,j0+1:1
        # v: j0,k0:1; j0+1,k0+1:1
        # w: k0,i0:1; k0+1,i0+1:1
        # Note: W index is k*n+i.
        
        # We need to add contributions to U, V, W for each non-zero entry.
        # Since U, V, W are sparse in the term dimension, we just set the values.
        
        # T1
        U[i0*m + j0, rank] = 1
        U[(i0+1)*m + (j0+1), rank] = 1
        V[j0*p + k0, rank] = 1
        V[(j0+1)*p + (k0+1), rank] = 1
        W[k0*n + i0, rank] = 1
        W[(k0+1)*n + (i0+1), rank] = 1
        rank += 1
        
        # T2: (A10+A11)B00
        U[i0*m + (j0+1), rank] = 1 # A10? No, A_{1,0} is i0+1, j0.
        # Wait, A is 2x2.
        # A00: i0,j0. A01: i0,j0+1.
        # A10: i0+1,j0. A11: i0+1,j0+1.
        
        # T2: (A10+A11) * B00
        # u: i0+1,j0:1; i0+1,j0+1:1
        # v: j0,k0:1
        # w: k0,i0+1:1? No, C10 = M2 + M4.
        # Let's stick to the standard Strassen coefficients.
        
        # I will hardcode the 7 terms for the 2x2x2 block relative to local indices 0,1.
        
    # This is getting complex to implement dynamically with offsets.
    # I will generate the full decomposition for one 2x2x2 block and then shift indices.
    
    # Let's define the 7 rank-1 tensors for 2x2x2.
    # Local indices: i in 0,1; j in 0,1; k in 0,1.
    # U_local shape (2*2, 7), V_local (2*2, 7), W_local (2*2, 7).
    
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))
    
    # T1: (A00+A11)(B00+B11)
    U_loc[0, 0] = 1 # A00
    U_loc[3, 0] = 1 # A11
    V_loc[0, 0] = 1 # B00
    V_loc[3, 0] = 1 # B11
    W_loc[0, 0] = 1 # C00 (k0*i0) -> 0*2+0
    W_loc[3, 0] = 1 # C11 (k1*i1) -> 1*2+1
    
    # T2: (A10+A11)B00
    U_loc[2, 1] = 1 # A10
    U_loc[3, 1] = 1 # A11
    V_loc[0, 1] = 1 # B00
    W_loc[2, 1] = 1 # C10 (k0*i1) -> 0*2+1
    
    # T3: A00(B10-B11)
    U_loc[0, 2] = 1 # A00
    V_loc[2, 2] = 1 # B10
    V_loc[3, 2] = -1 # B11
    W_loc[1, 2] = 1 # C01 (k1*i0) -> 1*2+0
    
    # T4: A11(B01-B00)
    U_loc[3, 3] = 1 # A11
    V_loc[1, 3] = 1 # B01
    V_loc[0, 3] = -1 # B00
    W_loc[2, 3] = 1 # C10
    W_loc[0, 3] = 1 # C00
    
    # T5: (A00+A01)B11
    U_loc[0, 4] = 1 # A00
    U_loc[1, 4] = 1 # A01
    V_loc[3, 4] = 1 # B11
    W_loc[1, 4] = 1 # C01
    W_loc[3, 4] = -1 # C11
    
    # T6: (A10-A00)(B00+B10)
    U_loc[2, 5] = 1 # A10
    U_loc[0, 5] = -1 # A00
    V_loc[0, 5] = 1 # B00
    V_loc[2, 5] = 1 # B10
    W_loc[2, 5] = 1 # C11
    
    # T7: (A01-A11)(B11+B01)
    U_loc[1, 6] = 1 # A01
    U_loc[3, 6] = -1 # A11
    V_loc[3, 6] = 1 # B11
    V_loc[1, 6] = 1 # B01
    W_loc[3, 6] = 1 # C11
    W_loc[1, 6] = -1 # C01
    
    # Now map these to global indices for each block.
    
    # Blocks for j in {0,1}
    # k in {0,1}, {2,3}, {4,5}, {6,7}, {8,9}
    
    for j0 in [0, 1]:
        # Wait, Strassen requires j to be the inner dimension of size 2.
        # Our j dimension is 3.
        # We split j into {0,1} and {2}.
        # For j in {0,1}, we use Strassen.
        # But Strassen acts on 2x2 matrices.
        # Here A is 2x2 (rows 0,1; cols 0,1).
        # B is 2x2 (rows 0,1; cols k0, k0+1).
        # So we need to iterate over k blocks.
        
        # The j index in the global U/V is fixed?
        # No, U index is i*m+j.
        # For a specific block, we use specific i, j, k.
        # But Strassen combines different j's?
        # No, Strassen for 2x2x2 uses j=0 and j=1.
        # So for each k-block, we use j=0 and j=1.
        
        for k0 in [0, 2, 4, 6, 8]:
            # i0 is 0.
            i0 = 0
            # j0 is 0.
            # But we have two j's: 0 and 1.
            # The block covers i in {0,1}, j in {0,1}, k in {k0, k0+1}.
            
            # We add the 7 terms.
            for r in range(7):
                # Map local U indices (i*m+j) to global
                # Local U index l corresponds to i=l//2, j=l%2?
                # No, U_loc is 4x7.
                # Index 0: i=0, j=0.
                # Index 1: i=0, j=1.
                # Index 2: i=1, j=0.
                # Index 3: i=1, j=1.
                
                # Global U index = i*3 + j.
                # Global V index = j*10 + k.
                # Global W index = k*2 + i.
                
                # Iterate over non-zero entries in U_loc[:, r]
                for lu in range(4):
                    if U_loc[lu, r] != 0:
                        li = lu // 2
                        lj = lu % 2
                        gu = li * m + lj
                        U[gu, rank] += U_loc[lu, r]
                        
                for lv in range(4):
                    if V_loc[lv, r] != 0:
                        lj = lv // 2
                        lk = lv % 2
                        # Global k is k0 + lk
                        gv = lj * p + (k0 + lk)
                        V[gv, rank] += V_loc[lv, r]
                        
                for lw in range(4):
                    if W_loc[lw, r] != 0:
                        lk = lw // 2
                        li = lw % 2
                        # Global k is k0 + lk
                        gw = (k0 + lk) * n + li
                        W[gw, rank] += W_loc[lw, r]
                        
                rank += 1

    # Now handle j=2
    # Trivial decomposition for A_{i,2} B_{2,k}
    # i in 0,1; k in 0..9
    for i in range(n):
        for k in range(p):
            # U index: i*m + 2
            # V index: 2*p + k
            # W index: k*n + i
            U[i*m + 2, rank] = 1
            V[2*p + k, rank] = 1
            W[k*n + i, rank] = 1
            rank += 1
            
    # Trim to actual rank
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
