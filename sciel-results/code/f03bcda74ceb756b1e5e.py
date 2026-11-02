"""Matrix multiplication 4,5,5 with block Strassen-based rank minimization."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the matmul tensor and check reconstruction
    # T[i,j,k] = delta (output element C[i,k] depends on A[i,j]*B[j,k])
    # For each term r: sum over i,j,k of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should match the identity
    # More precisely, the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for valid indices
    # And U @ outer(V[:,r], W[:,r]) summed over r should reproduce T
    # Let's just check with random matrices
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        # Reconstruct C from decomposition
        C_rec = np.zeros((n, p))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # A_part[i,j] = u_col[i*m+j], B_part[j,k] = v_col[j*p+k]
            A_part = u_col.reshape(n, m)
            B_part = v_col.reshape(m, p)
            C_part = A_part @ B_part
            # W maps C_part to output: w_col[k*n+i]
            # C_rec[i,k] += sum over ... actually C_rec[i,k] should get contribution
            # The decomposition says: C[i,k] = sum_r w_col[k*n+i] * (A_part @ B_part)[i,k]
            # Wait, that's not right. Let me re-think.
            # Standard: C[i,k] = sum_j A[i,j] B[j,k]
            # Decomposition: C[i,k] = sum_r U[i,m+j... no
            # U has shape (n*m, rank), indexed by [i*m + j, r]
            # V has shape (m*p, rank), indexed by [j*p + k, r]  
            # W has shape (p*n, rank), indexed by [k*n + i, r]
            # The tensor identity is: sum_r U[i*m+j, r] * V[j*p+k, r] = delta_{...} 
            # Actually the standard formulation:
            # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * ... 
            # No. Let me use the standard definition.
            # The matmul tensor T is a 3-tensor. A rank-r decomposition gives
            # T = sum_r u_r ⊗ v_r ⊗ w_r where
            # u_r[i,j] = U[i*m+j, r], v_r[j,k] = V[j*p+k, r], w_r[i,k] = W[k*n+i, r]
            # And T[i,j,k] = 1 (it's just the identity mapping for each (i,j,k) -> C[i,k])
            # Actually T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
            # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if this is a valid (i,j,k) triple
            # But every (i,j,k) with i in [0,n), j in [0,m), k in [0,p) is valid.
            # Hmm, but that would mean sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for ALL i,j,k.
            # That's a very strong condition. Let me verify with schoolbook.
            # Schoolbook: for each (i,j,k), there's exactly one r where U,V,W are all 1.
            # So sum = 1. Correct.
            # For Strassen, the sum might not be exactly 1 for each (i,j,k) individually,
            # but when contracted with A and B it gives the right C.
            # Let me just do the actual matrix multiplication check.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # From decomposition: C[i,k] = sum_r W[k*n+i,r] * (sum_j A[i,j] * ?) 
            # Actually the correct interpretation:
            # Define for each r: 
            #   a_r = sum_{i,j} U[i*m+j,r] A[i,j]  -- no, that's a scalar
            # That's not right either. Let me think again.
            # 
            # The standard tensor decomposition for matmul:
            # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
            # where U[i,j,r] = U[i*m+j, r], V[j,k,r] = V[j*p+k, r], and W is... 
            # Wait, in the standard formulation there are only 2 factors per term, not 3.
            # Let me re-read the problem. "U, V, W factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
            # So it's a 3-way tensor decomposition. The matmul tensor T has indices (i*m+j, j*p+k, k*n+i)
            # and T[a,b,c] = 1 when a=i*m+j, b=j*p+k, c=k*n+i for valid i,j,k, else 0.
            # Rank-r decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
            # where u_r[a] = U[a,r], v_r[b] = V[b,r], w_r[c] = W[c,r]
            # So T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For this to equal the matmul tensor, we need:
            # For all valid (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
            # For all invalid triples: sum_r ... = 0
            # 
            # To check correctness, we can verify that for random A, B:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # equals
            # C'[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] A[i,j]) * (sum_j V[j*p+k,r] B[j,k])
            # Hmm, that's a specific contraction. Let me just verify by building the full tensor.
            pass
    # Direct tensor verification
    T_recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    if not np.allclose(T_recon, T_expected, atol=tol):
        diff = np.max(np.abs(T_recon - T_expected))
        return False, diff
    return True, 0.0

def _strassen_222():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    n = m = p = 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    # Strassen formulas:
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
    # A indices: A11=0, A12=1, A21=2, A22=3 (i*m+j: i=0,1; j=0,1)
    # B indices: B11=0, B12=1, B21=2, B22=3 (j*p+k: j=0,1; k=0,1)
    # C indices: C11=0, C12=1, C21=2, C22=3 (k*n+i: k=0,1; i=0,1)
    
    # M1: (A11+A22)(B11+B22) -> C11+=1, C22+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11 -> C21+=1, C22+=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A11(B21-B22) -> C12+=1, C22+=1
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A22(B12-B11) -> C11+=1, C21+=1
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A11+A12)B22 -> C11+=-1, C12+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A21-A11)(B11+B12) -> C22+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22) -> C11+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_221():
    """Return U, V, W for 2x2x1 schoolbook (rank 4)."""
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((2, rank))
    W = np.zeros((2, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _schoolbook_215():
    """Return U, V, W for 2x1x5 schoolbook (rank 10)."""
    n, m, p = 2, 1, 5
    rank = 10
    U = np.zeros((2, rank))
    V = np.zeros((5, rank))
    W = np.zeros((10, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _schoolbook_225():
    """Return U, V, W for 2x2x5 schoolbook (rank 20)."""
    n, m, p = 2, 2, 5
    rank = 20
    U = np.zeros((4, rank))
    V = np.zeros((10, rank))
    W = np.zeros((10, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _combine_blocks(n_blocks, U_blocks, V_blocks, W_blocks, dims):
    """
    Combine block decompositions into a full decomposition.
    dims = [(n, m, p), ...] for each block
    U_blocks, V_blocks, W_blocks are lists of (U, V, W) arrays for each block.
    """
    # The blocks correspond to partitions of the matrix dimensions.
    # We need to map local indices to global indices.
    # For simplicity, we'll handle specific cases.
    raise NotImplementedError("Use explicit assembly functions")

def _assemble_455():
    """
    Assemble a rank-minimal decomposition for 4x5x5 matrix multiplication.
    
    Strategy: Split 4x5 into (2+2)x5.
    C = C_top + C_bottom where C_top = A_top * B, C_bottom = A_bottom * B
    A_top is 2x5, A_bottom is 2x5.
    
    For 2x5x5: Split 5 into (2+2+1) or use other decompositions.
    
    Let's try: 2x5x5 decomposition.
    Split the 5 columns of B (and rows of C) into parts of size 2, 2, 1.
    C_top = A_top * B = [A_top*B1, A_top*B2, A_top*B3]
    where B1 is 5x2, B2 is 5x2, B3 is 5x1.
    Wait, B is 5x5. A_top is 2x5.
    C_top[i,k] = sum_j A_top[i,j] B[j,k]
    
    Split j into groups: j in {0,1}, j in {2,3}, j in {4}.
    C_top = A_top[:,0:2] @ B[0:2,:] + A_top[:,2:4] @ B[2:4,:] + A_top[:,4:5] @ B[4:5,:]
    
    Each term:
    Term 1: (2x2) @ (2x5) -> 2x5. This is 2x2x5, rank 20 (schoolbook) or better.
    Term 2: (2x2) @ (2x5) -> 2x5. Same, rank 20.
    Term 3: (2x1) @ (1x5) -> 2x5. This is 2x1x5, rank 10.
    
    So 2x5x5 has rank at most 20+20+10 = 50.
    Similarly for the bottom 2x5x5, another 50.
    Total rank = 100. Same as schoolbook! Not helpful.
    
    Better approach: Use Strassen on the 2x2 blocks.
    For 2x2x2 we have rank 7 instead of 8.
    
    Let me think of a better decomposition for 4x5x5.
    
    Alternative: Split 4x5x5 as (2+2)x(5)x(5).
    C = [A1; A2] @ B where A1 is 2x5, A2 is 2x5.
    C1 = A1 @ B (2x5)
    C2 = A2 @ B (2x5)
    
    For each 2x5x5, split the inner dimension (5) into (2+2+1):
    A1 @ B = A1[:,0:2]@B[0:2,:] + A1[:,2:4]@B[2:4,:] + A1[:,4:5]@B[4:5,:]
    
    For the (2x2)@(2x5) parts, we can use a Strassen-like approach.
    Actually, (2x2)@(2x5) is like multiplying two 2x2 blocks but the second has 5 columns.
    We can view B[0:2,:] as a 2x5 matrix. Split its columns into (2+2+1):
    A1[:,0:2] @ B[0:2,:] = A1[:,0:2] @ [B11, B12, B13] where B11 is 2x2, B12 is 2x2, B13 is 2x1.
    
    So A1[:,0:2] @ B11 is 2x2x2 (rank 7 with Strassen)
    A1[:,0:2] @ B12 is 2x2x2 (rank 7)
    A1[:,0:2] @ B13 is 2x2x1 (rank 4)
    
    Similarly for A1[:,2:4]:
    A1[:,2:4] @ B[2:4,:] where B[2:4,:] has columns split into (2+2+1):
    B[2:4,0:2] is 2x2, B[2:4,2:4] is 2x2, B[2:4,4:5] is 2x1
    Each: 2x2x2 (rank 7), 2x2x2 (rank 7), 2x2x1 (rank 4)
    
    And A1[:,4:5] @ B[4:5,:] is 2x1x5 (rank 10)
    
    Total for one 2x5x5: 7+7+4+7+7+4+10 = 46
    Two such: 92
    
    Hmm, 92 < 100. That's an improvement!
    
    Let me be more careful. Actually, I think I can do even better by using the full Strassen structure.
    
    Let me just implement the 92-rank decomposition and verify it.
    """
    n, m, p = 4, 5, 5
    rank = 92
    U = np.zeros((n*m, rank))  # (20, 92)
    V = np.zeros((m*p, rank))  # (25, 92)
    W = np.zeros((p*n, rank))  # (20, 92)
    
    col = 0
    
    def add_block(U_b, V_b, W_b, i_off, j_off, k_off):
        """Add a block decomposition to the global matrices.
        U_b has shape (n_b*m_b, r_b), indexed locally as [i_b*m_b + j_b, r_b]
        V_b has shape (m_b*p_b, r_b), indexed locally as [j_b*p_b + k_b, r_b]
        W_b has shape (p_b*n_b, r_b), indexed locally as [k_b*n_b + i_b, r_b]
        Global indices: i = i_off + i_b, j = j_off + j_b, k = k_off + k_b
        """
        nonlocal col
        r_b = U_b.shape[1]
        n_b = U_b.shape[0] // U_b.shape[1] if False else None
        # We need to know the block dimensions to map indices.
        # Let's pass them explicitly.
        raise Exception("Use explicit mapping")
    
    # This is getting complex. Let me just build it directly.
    
    # Decomposition for top half (i in 0,1) and bottom half (i in 2,3)
    for half in range(2):
        i_base = half * 2  # 0 for top, 2 for bottom
        
        # For each half, decompose A[i_base:i_base+2, :] @ B
        # A_half is 2x5, B is 5x5
        # Split columns of A_half (and rows of B) into groups of size 2, 2, 1
        
        # Group 1: j in 0,1
        for j_group in range(3):
            if j_group < 2:
                j_base = j_group * 2
                m_b = 2
            else:
                j_base = 4
                m_b = 1
            
            # Split columns of B (and output C) into groups of size 2, 2, 1
            for k_group in range(3):
                if k_group < 2:
                    k_base = k_group * 2
                    p_b = 2
                else:
                    k_base = 4
                    p_b = 1
                
                # We need a decomposition for (2 x m_b x p_b)
                # m_b is 2 or 1, p_b is 2 or 1
                
                if m_b == 2 and p_b == 2:
                    # 2x2x2: Strassen, rank 7
                    Ub, Vb, Wb = _strassen_222()
                    # Map local indices to global
                    # Local A: [i_b, j_b] -> global [i_base+i_b, j_base+j_b] -> index (i_base+i_b)*5 + (j_base+j_b)
                    # Local B: [j_b, k_b] -> global [j_base+j_b, k_base+k_b] -> index (j_base+j_b)*5 + (k_base+k_b)
                    # Local C: [k_b, i_b] -> global [k_base+k_b, i_base+i_b] -> index (k_base+k_b)*4 + (i_base+i_b)
                    for r_b in range(7):
                        for i_b in range(2):
                            for j_b in range(2):
                                if Ub[i_b*2+j_b, r_b] != 0:
                                    U[(i_base+i_b)*5 + (j_base+j_b), col] += Ub[i_b*2+j_b, r_b]
                        for j_b in range(2):
                            for k_b in range(2):
                                if Vb[j_b*2+k_b, r_b] != 0:
                                    V[(j_base+j_b)*5 + (k_base+k_b), col] += Vb[j_b*2+k_b, r_b]
                        for k_b in range(2):
                            for i_b in range(2):
                                if Wb[k_b*2+i_b, r_b] != 0:
                                    W[(k_base+k_b)*4 + (i_base+i_b), col] += Wb[k_b*2+i_b, r_b]
                        col += 1
                    
                elif m_b == 2 and p_b == 1:
                    # 2x2x1: schoolbook, rank 4
                    Ub, Vb, Wb = _schoolbook_221()
                    for r_b in range(4):
                        for i_b in range(2):
                            for j_b in range(2):
                                if Ub[i_b*2+j_b, r_b] != 0:
                                    U[(i_base+i_b)*5 + (j_base+j_b), col] += Ub[i_b*2+j_b, r_b]
                        for j_b in range(2):
                            for k_b in range(1):
                                if Vb[j_b*1+k_b, r_b] != 0:
                                    V[(j_base+j_b)*5 + (k_base+k_b), col] += Vb[j_b*1+k_b, r_b]
                        for k_b in range(1):
                            for i_b in range(2):
                                if Wb[k_b*2+i_b, r_b] != 0:
                                    W[(k_base+k_b)*4 + (i_base+i_b), col] += Wb[k_b*2+i_b, r_b]
                        col += 1
                        
                elif m_b == 1 and p_b == 2:
                    # 2x1x2: schoolbook, rank 4
                    n_b, m_b_local, p_b_local = 2, 1, 2
                    rank_b = 4
                    Ub = np.zeros((n_b*m_b_local, rank_b))
                    Vb = np.zeros((m_b_local*p_b_local, rank_b))
                    Wb = np.zeros((p_b_local*n_b, rank_b))
                    c = 0
                    for i_b in range(n_b):
                        for j_b in range(m_b_local):
                            for k_b in range(p_b_local):
                                Ub[i_b*m_b_local+j_b, c] = 1
                                Vb[j_b*p_b_local+k_b, c] = 1
                                Wb[k_b*n_b+i_b, c] = 1
                                c += 1
                    for r_b in range(4):
                        for i_b in range(2):
                            for j_b in range(1):
                                if Ub[i_b*1+j_b, r_b] != 0:
                                    U[(i_base+i_b)*5 + (j_base+j_b), col] += Ub[i_b*1+j_b, r_b]
                        for j_b in range(1):
                            for k_b in range(2):
                                if Vb[j_b*2+k_b, r_b] != 0:
                                    V[(j_base+j_b)*5 + (k_base+k_b), col] += Vb[j_b*2+k_b, r_b]
                        for k_b in range(2):
                            for i_b in range(2):
                                if Wb[k_b*2+i_b, r_b] != 0:
                                    W[(k_base+k_b)*4 + (i_base+i_b), col] += Wb[k_b*2+i_b, r_b]
                        col += 1
                        
                elif m_b == 1 and p_b == 1:
                    # 2x1x1: schoolbook, rank 2
                    n_b, m_b_local, p_b_local = 2, 1, 1
                    rank_b = 2
                    Ub = np.zeros((n_b*m_b_local, rank_b))
                    Vb = np.zeros((m_b_local*p_b_local, rank_b))
                    Wb = np.zeros((p_b_local*n_b, rank_b))
                    c = 0
                    for i_b in range(n_b):
                        for j_b in range(m_b_local):
                            for k_b in range(p_b_local):
                                Ub[i_b*m_b_local+j_b, c] = 1
                                Vb[j_b*p_b_local+k_b, c] = 1
                                Wb[k_b*n_b+i_b, c] = 1
                                c += 1
                    for r_b in range(2):
                        for i_b in range(2):
                            for j_b in range(1):
                                if Ub[i_b*1+j_b, r_b] != 0:
                                    U[(i_base+i_b)*5 + (j_base+j_b), col] += Ub[i_b*1+j_b, r_b]
                        for j_b in range(1):
                            for k_b in range(1):
                                if Vb[j_b*1+k_b, r_b] != 0:
                                    V[(j_base+j_b)*5 + (k_base+k_b), col] += Vb[j_b*1+k_b, r_b]
                        for k_b in range(1):
                            for i_b in range(2):
                                if Wb[k_b*2+i_b, r_b] != 0:
                                    W[(k_base+k_b)*4 + (i_base+i_b), col] += Wb[k_b*2+i_b, r_b]
                        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def construct():
    U, V, W = _assemble_455()
    # Convert to lists for the expected interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
