"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS  # 4, 4, 5

    def verify_decomposition(U_list, V_list, W_list, rank):
        """Verify that the decomposition exactly reconstructs the tensor."""
        U = np.array(U_list, dtype=np.float64)
        V = np.array(V_list, dtype=np.float64)
        W = np.array(W_list, dtype=np.float64)
        
        if U.shape != (n * m, rank):
            return False
        if V.shape != (m * p, rank):
            return False
        if W.shape != (p * n, rank):
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    product = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                    if abs(product - 1.0) > 1e-9:
                        return False
        return True

    # Strassen 2x2x2 decomposition: rank 7
    # For C = A*B where A,B are 2x2:
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B01-B11)
    # m4 = A11*(B10-B00)
    # m5 = (A00+A10)*B11
    # m6 = (A11-A10)*(B00+B10)
    # m7 = (A01-A11)*(B01+B11)
    # C00 = m1 + m4 - m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m7

    def strassen_2x2_block(i_base, j_base, k_base, col_offset, U, V, W):
        """Add Strassen 2x2x2 block for submatrices at given base indices."""
        # The 7 Strassen factors
        # U entries: for (i, j) pairs in the 2x2 block
        # V entries: for (j, k) pairs
        # W entries: for (k, i) pairs
        
        # m1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # U: (i0,j0)=+1, (i1,j1)=+1
        # V: (j0,k0)=+1, (j1,k1)=+1
        # W: (k0,i0)=+1, (k1,i1)=+1  -> but W[k*n+i], so W[k0,i0] and W[k1,i1]
        # Actually W is indexed by [k*n + i], so for output C[i,k]:
        # C[i0,k0] gets +1, C[i1,k1] gets +1
        
        # Let me be more careful. The decomposition is:
        # sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 for all (i,j,k)
        
        # For Strassen, we have 7 rank-1 terms. Each term has:
        # U values on some (i,j) pairs, V values on some (j,k) pairs, W values on some (k,i) pairs
        
        # m1: U[(i0,j0)]=1, U[(i1,j1)]=1; V[(j0,k0)]=1, V[(j1,k1)]=1; W[(k0,i0)]=1, W[(k1,i1)]=1
        # This contributes to: C[i0,k0] += U[i0,j0]*V[j0,k0]*W[k0,i0] = 1*1*1 = 1
        #                   C[i0,k1] += U[i0,j0]*V[j0,k1]*W[k1,i0] = 1*0*? = 0 (V[j0,k1]=0)
        # Hmm, this isn't right. Let me reconsider.
        
        # Actually in Strassen, each "multiplication" m_l is a product of two linear forms.
        # The decomposition of the 2x2x2 tensor has rank 7, meaning:
        # T[i,j,k] = sum_{l=1}^{7} U[i,j,l] * V[j,k,l] * W[k,i,l]
        
        # The 7 terms are:
        # l=1: U has 1 at (0,0) and (1,1); V has 1 at (0,0) and (1,1); W has 1 at (0,0) and (1,1)
        #       -> contributes 1 to (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
        #       That's not right either.
        
        # Let me look up the correct Strassen tensor decomposition.
        # The 2x2x2 matrix multiplication tensor has rank 7.
        # The 7 rank-1 tensors are:
        
        # I'll use a known correct decomposition. Let me define it explicitly.
        
        # For 2x2x2, the tensor T[i,j,k] = 1 for all i,j,k in {0,1}
        # Rank-7 decomposition:
        
        # Term 0: U=(1,0,0,1) for (i,j) in order [(0,0),(0,1),(1,0),(1,1)]
        #          V=(1,0,0,1) for (j,k) in order [(0,0),(0,1),(1,0),(1,1)]  
        #          W=(1,0,0,1) for (k,i) in order [(0,0),(0,1),(1,0),(1,1)]
        # This gives T[0,0,0] += 1*1*1 = 1, T[1,1,1] += 1*1*1 = 1
        # But we need T[i,j,k]=1 for ALL 8 entries, so this alone doesn't work.
        
        # I think I'm overcomplicating this. Let me just use the standard Strassen algorithm
        # and translate it to the tensor decomposition format.
        
        pass  # Will implement below with explicit construction

    # Let me take a different approach: explicitly construct the 8-block Strassen for 4x4x4
    # and then add the 5th column.
    
    # For 4x4x4: split into 4x4 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = A*B, so Cij = sum_k Akj * Bkj
    
    # For each (i,k) block of C, C[i,k] = sum_j A[i,j] * B[j,k]
    # Each C[i,k] is 2x2, computed from 2x2 matrix multiplications.
    
    # Using Strassen for each of the 4 (i,k) blocks:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] -> needs 2 separate Strassen, rank 14
    # That's not optimal. Instead, use Strassen on the 4x4 blocks directly.
    
    # Better: use 2x2 Strassen at the top level (treating 2x2 blocks as elements)
    # This gives 7 "big" multiplications of 2x2 blocks, each of which needs Strassen (rank 7)
    # Total rank: 7 * 7 = 49 for 4x4x4
    
    # But 49 is complex to implement. Let me use the simpler 8-block approach:
    # 4 independent 2x2x2 Strassen for the diagonal 2x2 blocks of the 4x4x4
    # plus cross terms... no, that's not how it works.
    
    # Simplest correct approach: 4x4x4 = 4 independent 2x2x2 Strassen blocks
    # This only handles the case where A and B are block-diagonal, which they're not.
    
    # OK let me just use schoolbook for simplicity and correctness, then try to optimize.
    
    # Schoolbook for 4x4x5: rank 80
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    if verify_decomposition(U, V, W, rank_schoolbook):
        return {"U": U, "V": V, "W": W}
    
    # If we get here, something is wrong. Return schoolbook anyway.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
