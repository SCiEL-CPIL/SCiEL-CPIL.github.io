import numpy as np

def test_laderman():
    n = m = p = 4
    rank = 19
    
    # Define U, V for the 19 terms
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    
    # Term 0: A[0,0]*B[0,0]
    U[0*4+0, 0] = 1
    V[0*4+0, 0] = 1
    
    # Term 1: A[1,1]*B[1,1]
    U[1*4+1, 1] = 1
    V[1*4+1, 1] = 1
    
    # Term 2: A[2,2]*B[2,2]
    U[2*4+2, 2] = 1
    V[2*4+2, 2] = 1
    
    # Term 3: A[3,3]*B[3,3]
    U[3*4+3, 3] = 1
    V[3*4+3, 3] = 1
    
    # Term 4: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0*4+0, 4] = 1
    U[1*4+1, 4] = 1
    V[0*4+0, 4] = 1
    V[1*4+1, 4] = 1
    
    # Term 5: (A[0,0]+A[2,2])*(B[0,0]+B[2,2])
    U[0*4+0, 5] = 1
    U[2*4+2, 5] = 1
    V[0*4+0, 5] = 1
    V[2*4+2, 5] = 1
    
    # Term 6: (A[0,0]+A[3,3])*(B[0,0]+B[3,3])
    U[0*4+0, 6] = 1
    U[3*4+3, 6] = 1
    V[0*4+0, 6] = 1
    V[3*4+3, 6] = 1
    
    # Term 7: (A[1,1]+A[2,2])*(B[1,1]+B[2,2])
    U[1*4+1, 7] = 1
    U[2*4+2, 7] = 1
    V[1*4+1, 7] = 1
    V[2*4+2, 7] = 1
    
    # Term 8: (A[1,1]+A[3,3])*(B[1,1]+B[3,3])
    U[1*4+1, 8] = 1
    U[3*4+3, 8] = 1
    V[1*4+1, 8] = 1
    V[3*4+3, 8] = 1
    
    # Term 9: (A[2,2]+A[3,3])*(B[2,2]+B[3,3])
    U[2*4+2, 9] = 1
    U[3*4+3, 9] = 1
    V[2*4+2, 9] = 1
    V[3*4+3, 9] = 1
    
    # Term 10: (A[0,1]+A[1,0])*(B[1,0]+B[0,1])
    U[0*4+1, 10] = 1
    U[1*4+0, 10] = 1
    V[1*4+0, 10] = 1
    V[0*4+1, 10] = 1
    
    # Term 11: (A[0,2]+A[2,0])*(B[2,0]+B[0,2])
    U[0*4+2, 11] = 1
    U[2*4+0, 11] = 1
    V[2*4+0, 11] = 1
    V[0*4+2, 11] = 1
    
    # Term 12: (A[0,3]+A[3,0])*(B[3,0]+B[0,3])
    U[0*4+3, 12] = 1
    U[3*4+0, 12] = 1
    V[3*4+0, 12] = 1
    V[0*4+3, 12] = 1
    
    # Term 13: (A[1,2]+A[2,1])*(B[2,1]+B[1,2])
    U[1*4+2, 13] = 1
    U[2*4+1, 13] = 1
    V[2*4+1, 13] = 1
    V[1*4+2, 13] = 1
    
    # Term 14: (A[1,3]+A[3,1])*(B[3,1]+B[1,3])
    U[1*4+3, 14] = 1
    U[3*4+1, 14] = 1
    V[3*4+1, 14] = 1
    V[1*4+3, 14] = 1
    
    # Term 15: (A[2,3]+A[3,2])*(B[3,2]+B[2,3])
    U[2*4+3, 15] = 1
    U[3*4+2, 15] = 1
    V[3*4+2, 15] = 1
    V[2*4+3, 15] = 1
    
    # Term 16: (A[0,1]+A[1,0]+A[0,2]+A[2,0])*(B[1,0]+B[0,1]+B[2,0]+B[0,2])
    U[0*4+1, 16] = 1
    U[1*4+0, 16] = 1
    U[0*4+2, 16] = 1
    U[2*4+0, 16] = 1
    V[1*4+0, 16] = 1
    V[0*4+1, 16] = 1
    V[2*4+0, 16] = 1
    V[0*4+2, 16] = 1
    
    # Term 17: (A[0,3]+A[3,0]+A[1,2]+A[2,1])*(B[3,0]+B[0,3]+B[2,1]+B[1,2])
    U[0*4+3, 17] = 1
    U[3*4+0, 17] = 1
    U[1*4+2, 17] = 1
    U[2*4+1, 17] = 1
    V[3*4+0, 17] = 1
    V[0*4+3, 17] = 1
    V[2*4+1, 17] = 1
    V[1*4+2, 17] = 1
    
    # Term 18: (A[1,3]+A[3,1]+A[2,3]+A[3,2])*(B[3,1]+B[1,3]+B[3,2]+B[2,3])
    U[1*4+3, 18] = 1
    U[3*4+1, 18] = 1
    U[2*4+3, 18] = 1
    U[3*4+2, 18] = 1
    V[3*4+1, 18] = 1
    V[1*4+3, 18] = 1
    V[3*4+2, 18] = 1
    V[2*4+3, 18] = 1
    
    # Now solve for W. For each (i', k'), we need W[k'*4+i', r] such that:
    # Σ_r U[i*4+j, r] * V[j'*4+k, r] * W[k'*4+i', r] = δ_{j,j'} * δ_{i,i'} * δ_{k,k'}
    # for all i, j, j', k.
    
    # For a fixed (i', k'), this is a linear system in W[k'*4+i', r].
    # But it's overdetermined (256 equations, 19 unknowns).
    # Let me just check if the system is consistent.
    
    # Actually, let me just try to solve for W using least squares and check the residual.
    
    W = np.zeros((16, rank))
    
    for i_prime in range(4):
        for k_prime in range(4):
            # Build the system: for all i, j, j', k:
            # Σ_r U[i*4+j, r] * V[j'*4+k, r] * W[k'*4+i', r] = δ_{j,j'} * δ_{i,i_prime} * δ_{k,k_prime}
            
            # This is a linear system in W[k'*4+i', r] for r=0..18.
            # Number of equations: 4*4*4*4 = 256
            # Number of unknowns: 19
            
            # Build the matrix A_sys (256 x 19) and vector b (256)
            A_sys = np.zeros((256, rank))
            b = np.zeros(256)
            
            eq_idx = 0
            for i in range(4):
                for j in range(4):
                    for j_prime in range(4):
                        for k in range(4):
                            for r in range(rank):
                                A_sys[eq_idx, r] = U[i*4+j, r] * V[j_prime*4+k, r]
                            b[eq_idx] = (1 if j == j_prime else 0) * (1 if i == i_prime else 0) * (1 if k == k_prime else 0)
                            eq_idx += 1
            
            # Solve least squares
            W_row, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b, rcond=None)
            W[k_prime*4+i_prime, :] = W_row
    
    # Verify
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_test = A_test @ B_test
    
    # Compute using decomposition
    C_decomp = np.zeros((4, 4))
    for r in range(rank):
        C_decomp += (A_test @ (U[:, r].reshape(4,4))) * (V[:, r].reshape(4,4) @ B_test.T).T * W[:, r].reshape(4,4)
    
    # Wait, that's not right. Let me think about how to compute C from the decomposition.
    # C[i',k'] = Σ_r W[k'*4+i', r] * (Σ_j U[i'*4+j, r] * A[i',j]) * (Σ_{j'} V[j'*4+k', r] * B[j',k'])
    
    C_decomp = np.zeros((4, 4))
    for i_prime in range(4):
        for k_prime in range(4):
            val = 0.0
            for r in range(rank):
                a_prod = sum(U[i_prime*4+j, r] * A_test[i_prime, j] for j in range(4))
                b_prod = sum(V[j_prime*4+k_prime, r] * B_test[j_prime, k_prime] for j_prime in range(4))
                val += W[k_prime*4+i_prime, r] * a_prod * b_prod
            C_decomp[i_prime, k_prime] = val
    
    print("Max error:", np.max(np.abs(C_test - C_decomp)))
    print("Residual:", residuals)
    
    return U, V, W
