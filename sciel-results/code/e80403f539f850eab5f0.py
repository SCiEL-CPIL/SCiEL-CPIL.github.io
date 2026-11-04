import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    
    # We'll use a constructive decomposition:
    # A (2x3) split into A1 (2x2, cols 0-1) and A2 (2x1, col 2)
    # B (3x4) split into B1 (2x4, rows 0-1) and B2 (1x4, row 2)
    # C = A1 @ B1 + A2 @ B2
    # 
    # For A1 @ B1: B1 has 4 columns. Split into B1a (2x2, cols 0-1) and B1b (2x2, cols 2-3)
    # Apply Strassen to A1 @ B1a and A1 @ B1b: 7 + 7 = 14 terms
    # For A2 @ B2: outer product, rank 1
    # Total rank: 15
    
    # But wait - can we do better? Let me think about whether we can share Strassen terms.
    # Actually, the Strassen terms for A1@B1a and A1@B1b use different B blocks, so they're independent.
    
    # Let me try rank 15 first, then see if we can optimize.
    
    # Actually, let me reconsider the problem structure more carefully.
    # The tensor is T[i,j,k] = delta (for the multiplication tensor)
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,j,k} in the appropriate sense
    
    # Wait, I need to re-read the task. The output is W[k*n+i, r].
    # The decomposition is: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let me verify: for standard matrix multiplication C = A @ B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor element is T[i,j,k] = 1 (for all valid i,j,k)
    # And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # No wait, that's not right either. Let me re-read.
    # "the output factor follows the official trace convention W[k*n+i,r]"
    
    # I think the decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r (sum_j U[i*m+j,r] A[i,j]) (sum_j' V[j'*p+k,r] B[j',k]) ... no that doesn't work
    
    # Actually, I think it's:
    # C[i,k] = sum_r U[i*m+?, r] * V[?, r] * W[k*n+i, r]
    
    # Let me look at the schoolbook solution in the original code:
    # For each (i,j,k): U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # So: C[i,k] = sum_{j} A[i,j] B[j,k] = sum_j sum_col U[i*m+j,col] V[j*p+k,col] W[k*n+i,col] A[i,j] B[j,k]
    # Wait, that gives sum_j 1*1*1*A[i,j]*B[j,k] = sum_j A[i,j]*B[j,k]. Yes!
    
    # So the factorization is:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # No, that's not a proper bilinear form.
    
    # Let me think again. The standard tensor decomposition for matrix multiplication:
    # M_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # where M_{i,j,k} is the multiplication tensor (which is 1 for all valid indices in the "trivial" case)
    
    # Actually, I think the correct interpretation is:
    # The multiplication tensor T has T[i,j,k] = 1 for all i,j,k (in the appropriate ranges)
    # And we want T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # But that's not quite right for matrix multiplication. Let me re-examine.
    
    # For C = A @ B: C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map from A and B to C.
    # The tensor for this bilinear map has components T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k.
    
    # The rank decomposition: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k' V[j,k',r] B[j,k']) ... 
    
    # I'm confusing myself. Let me just follow the schoolbook pattern from the original code.
    
    # Schoolbook: for each (i,j,k), one rank term with:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    # This means: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = number of r where all three are 1 for that (i,j,k)
    # In schoolbook, each (i,j,k) has exactly one r, so the sum = 1 for each (i,j,k).
    
    # So the constraint is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 24 equations.
    
    # Now, for the constructive approach:
    # We want to find U, V, W of some rank R such that the above holds.
    
    # Let me try the block approach:
    # Split j into j=0,1 (for A1@B1) and j=2 (for A2@B2)
    
    # For j=0,1: we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] to contribute appropriately
    # For j=2: similar
    
    # Actually, the constraint is per (i,j,k) triple, not per output (i,k).
    # So we need 24 equations.
    
    # Let me try a direct constructive approach for rank 15.
    
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For our problem, the "A" matrix is 2x2 (rows i=0,1; cols j=0,1 for the first part)
    # The "B" matrix is 2x2 (rows j=0,1; cols k=0,1 for the first k-block)
    
    # Let me define rank terms for the first k-block (k=0,1):
    # M1: U[0,0,r]=1, U[1,1,r]=1, V[0,0,r]=1, V[1,1,r]=1, W[0,0,r]=1, W[1,0,r]=1, W[0,1,r]=1, W[1,1,r]=1
    # Wait, I need to be more careful.
    
    # The Strassen decomposition gives C11, C12, C21, C22 from A(2x2) and B(2x2).
    # In our tensor, the output is indexed by (k,i) for W.
    
    # Let me define:
    # For the sub-problem with i in {0,1}, j in {0,1}, k in {0,1}:
    # A_sub[i,j] corresponds to A[i,j] (i=0,1; j=0,1)
    # B_sub[j,k] corresponds to B[j,k] (j=0,1; k=0,1)
    # C_sub[i,k] = sum_j A[i,j] B[j,k]
    
    # Strassen gives:
    # C_sub[0,0] = M1 + M4 - M5 + M7
    # C_sub[0,1] = M3 + M5
    # C_sub[1,0] = M2 + M4
    # C_sub[1,1] = M1 - M2 + M3 + M6
    
    # In terms of the tensor, we need:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for each (i,j,k)
    
    # For the sub-problem, the relevant (i,j,k) triples are those with i in {0,1}, j in {0,1}, k in {0,1}.
    
    # Let me assign rank indices for the 7 Strassen terms of the first k-block:
    # r = 0: M1
    # r = 1: M2
    # r = 2: M3
    # r = 3: M4
    # r = 4: M5
    # r = 5: M6
    # r = 6: M7
    
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # U: U[0*m+0, 0] = 1 (i=0,j=0), U[1*m+1, 0] = 1 (i=1,j=1)
    # V: V[0*p+0, 0] = 1 (j=0,k=0), V[1*p+1, 0] = 1 (j=1,k=1)
    # W: W[0*n+0, 0] = 1 (k=0,i=0), W[1*n+1, 0] = 1 (k=1,i=1)
    # Wait, but M1 contributes to C[0,0] and C[1,1] (with +1).
    # C[0,0] gets +M1, C[1,1] gets +M1.
    # So W[0*n+0, 0] = 1 (for k=0,i=0) and W[1*n+1, 0] = 1 (for k=1,i=1).
    
    # M2 = (A[1,0]+A[1,1])B[0,0]
    # U: U[1*m+0, 1] = 1, U[1*m+1, 1] = 1
    # V: V[0*p+0, 1] = 1
    # W: M2 contributes to C[1,0] (+1) and C[1,1] (-1).
    # So W[0*n+1, 1] = 1 (k=0,i=1), W[1*n+1, 1] = -1 (k=1,i=1)
    
    # M3 = A[0,0](B[1,0]-B[1,1])
    # U: U[0*m+0, 2] = 1
    # V: V[1*p+0, 2] = 1, V[1*p+1, 2] = -1
    # W: M3 contributes to C[0,1] (+1) and C[1,1] (+1).
    # So W[1*n+0, 2] = 1 (k=1,i=0), W[1*n+1, 2] = 1 (k=1,i=1)
    
    # M4 = A[1,1](B[0,1]-B[0,0])
    # U: U[1*m+1, 3] = 1
    # V: V[0*p+1, 3] = 1, V[0*p+0, 3] = -1
    # W: M4 contributes to C[0,0] (+1) and C[1,0] (+1).
    # So W[0*n+0, 3] = 1, W[0*n+1, 3] = 1
    
    # M5 = (A[0,0]+A[0,1])B[1,1]
    # U: U[0*m+0, 4] = 1, U[0*m+1, 4] = 1
    # V: V[1*p+1, 4] = 1
    # W: M5 contributes to C[0,1] (+1) and C[0,0] (-1).
    # So W[1*n+0, 4] = 1, W[0*n+0, 4] = -1
    
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # U: U[1*m+0, 5] = 1, U[0*m+0, 5] = -1
    # V: V[0*p+0, 5] = 1, V[0*p+1, 5] = 1
    # W: M6 contributes to C[1,1] (+1).
    # So W[1*n+1, 5] = 1
    
    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    # U: U[0*m+1, 6] = 1, U[1*m+1, 6] = -1
    # V: V[1*p+0, 6] = 1, V[1*p+1, 6] = 1
    # W: M7 contributes to C[0,0] (+1).
    # So W[0*n+0, 6] = 1
    
    # Now for the second k-block (k=2,3), we apply the same Strassen pattern but with k offset by 2.
    # r = 7: M1'
    # r = 8: M2'
    # r = 9: M3'
    # r = 10: M4'
    # r = 11: M5'
    # r = 12: M6'
    # r = 13: M7'
    
    # And for j=2 (the A2@B2 part):
    # C[i,k] += A[i,2] * B[2,k]
    # This is an outer product. We need:
    # sum_r U[i*m+2, r] V[2*p+k, r] W[k*n+i, r] = 1 for each (i, k) with j=2.
    # Wait, the constraint is per (i,j,k) triple. For j=2:
    # We need sum_r U[i*m+2, r] V[2*p+k, r] W[k*n+i, r] = 1 for each i in {0,1}, k in {0,1,2,3}.
    
    # This is 4 equations. We can use rank 1:
    # r = 14: U[i*m+2, 14] = 1 for i=0,1; V[2*p+k, 14] = 1 for k=0,1,2,3; W[k*n+i, 14] = 1 for all k,i.
    # Check: sum_r U[i*m+2, r] V[2*p+k, r] W[k*n+i, r] = U[i*m+2,14]*V[2*p+k,14]*W[k*n+i,14] = 1*1*1 = 1. Correct!
    
    # But wait, we also need to make sure the Strassen terms don't interfere with j=2.
    # In the Strassen terms, U is only nonzero for j=0,1. V is only nonzero for j=0,1.
    # So for j=2, U[i*m+2, r] = 0 and V[2*p+k, r] = 0 for r=0..13.
    # Good, no interference.
    
    # Similarly, for j=0,1, we need to make sure the j=2 term doesn't interfere.
    # For j=0,1: U[i*m+j, 14] = 0 (since we only set U for j=2 at r=14).
    # V[j*p+k, 14] = 0 for j=0,1.
    # Good.
    
    # Total rank: 15.
    
    R = 15
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Helper to add a Strassen block
    def add_strassen(r_base, i_base, j_base, k_base):
        # M1
        U[i_base+0, j_base+0, r_base+0] += 1
        U[i_base+1, j_base+1, r_base+0] += 1
        V[j_base+0, k_base+0, r_base+0] += 1
        V[j_base+1, k_base+1, r_base+0] += 1
        W[k_base+0, i_base+0, r_base+0] += 1
        W[k_base+1, i_base+1, r_base+0] += 1
        
        # M2
        U[i_base+1, j_base+0, r_base+1] += 1
        U[i_base+1, j_base+1, r_base+1] += 1
        V[j_base+0, k_base+0, r_base+1] += 1
        W[k_base+0, i_base+1, r_base+1] += 1
        W[k_base+1, i_base+1, r_base+1] += -1
        
        # M3
        U[i_base+0, j_base+0, r_base+2] += 1
        V[j_base+1, k_base+0, r_base+2] += 1
        V[j_base+1, k_base+1, r_base+2] += -1
        W[k_base+1, i_base+0, r_base+2] += 1
        W[k_base+1, i_base+1, r_base+2] += 1
        
        # M4
        U[i_base+1, j_base+1, r_base+3] += 1
        V[j_base+0, k_base+1, r_base+3] += 1
        V[j_base+0, k_base+0, r_base+3] += -1
        W[k_base+0, i_base+0, r_base+3] += 1
        W[k_base+0, i_base+1, r_base+3] += 1
        
        # M5
        U[i_base+0, j_base+0, r_base+4] += 1
        U[i_base+0, j_base+1, r_base+4] += 1
        V[j_base+1, k_base+1, r_base+4] += 1
        W[k_base+1, i_base+0, r_base+4] += 1
        W[k_base+0, i_base+0, r_base+4] += -1
        
        # M6
        U[i_base+1, j_base+0, r_base+5] += 1
        U[i_base+0, j_base+0, r_base+5] += -1
        V[j_base+0, k_base+0, r_base+5] += 1
        V[j_base+0, k_base+1, r_base+5] += 1
        W[k_base+1, i_base+1, r_base+5] += 1
        
        # M7
        U[i_base+0, j_base+1, r_base+6] += 1
        U[i_base+1, j_base+1, r_base+6] += -1
        V[j_base+1, k_base+0, r_base+6] += 1
        V[j_base+1, k_base+1, r_base+6] += 1
        W[k_base+0, i_base+0, r_base+6] += 1
    
    # First k-block: k=0,1; i=0,1; j=0,1
    add_strassen(0, 0, 0, 0)
    # Second k-block: k=2,3; i=0,1; j=0,1
    add_strassen(7, 0, 0, 2)
    
    # j=2 part: outer product
    r = 14
    for i in range(n):
        U[i * m + 2, r] = 1
    for k in range(p):
        V[2 * p + k, r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, r] = 1
    
    # Verify
    exact = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0
                for r in range(R):
                    lhs += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if lhs != 1:
                    exact = False
                    break
            if not exact:
                break
        if not exact:
            break
    
    if exact:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    n, m, p = 2, 3, 4
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if lhs != 1:
                    print(f"FAIL: ({i},{j},{k}) = {lhs}")
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        print("Verification passed")
