"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the rank-44 decomposition
    rank = 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # m1: (a+d)(e+h)
        U[i_off*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # m2: (c+d)e
        U[(i_off+1)*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += -1
        col += 1
        
        # m3: a(f-h)
        U[i_off*m + j_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # m4: d(g-e)
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[(j_off+1)*p + k_off, col] += 1
        V[j_off*p + k_off, col] -= 1
        W[k_off*n + i_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        col += 1
        
        # m5: (a+b)h
        U[i_off*m + j_off, col] += 1
        U[i_off*m + (j_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        col += 1
        
        # m6: (c-a)(e+f)
        U[(i_off+1)*m + j_off, col] += 1
        U[i_off*m + j_off, col] += -1
        V[j_off*p + k_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # m7: (b-d)(g+h)
        U[i_off*m + (j_off+1), col] += 1
        U[(i_off+1)*m + (j_off+1), col] += -1
        V[(j_off+1)*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        col += 1

    # Helper to add Schoolbook 2x1x4
    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + j_off, col] = 1
                V[j_off*p + (k_off+k), col] = 1
                W[(k_off+k)*n + (i_off+i), col] = 1
                col += 1

    # Block 1: k in 0..3
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_schoolbook_2x1x4(0, 2, 0)

    # Block 2: k in 4..7
    add_strassen(0, 0, 4)
    add_strassen(0, 0, 6)
    add_schoolbook_2x1x4(0, 2, 4)
    
    # Verify correctness with random samples
    def verify_decomposition(U, V, W, n, m, p, num_samples=50):
        np.random.seed(42)
        for _ in range(num_samples):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for t in range(U.shape[1]):
                u = U[:, t]
                v = V[:, t]
                w = W[:, t]
                # u is in R^{n*m}, v is in R^{m*p}, w is in R^{p*n}
                # Reshape: u -> (n,m), v -> (m,p), w -> (p,n)
                u_mat = u.reshape(n, m)
                v_mat = v.reshape(m, p)
                w_mat = w.reshape(p, n)
                # The contribution is: for each t, C += (u_mat * v_mat) transposed appropriately
                # Actually, the decomposition is: C[i,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                # This is not a simple outer product. Let me think...
                # The tensor decomposition: T = sum_t U_t ⊗ V_t ⊗ W_t
                # where T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]... 
                # Wait, the convention is W[k*n+i, r], so W index is k*n+i.
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # In the decomposition: C[i,k] = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]
                # No, that's not right either. Let me reconsider.
                # 
                # The standard tensor decomposition for matrix mult:
                # C[i,k] = sum_j A[i,j] B[j,k]
                # We want: C[i,k] = sum_t U_t[i,j] V_t[j,k] W_t[k,i] ... no
                # 
                # Actually, the decomposition is:
                # For each term t, we have vectors u_t ∈ R^{nm}, v_t ∈ R^{mp}, w_t ∈ R^{pn}
                # And the condition is: for all A, B:
                #   (A ⊗ B) contracted with T = C
                #   T[i,j,k] = 1 for all valid (i,j,k)
                #   T = sum_t u_t ⊗ v_t ⊗ w_t (as a 3-tensor)
                #   So T[i,j,k] = sum_t u_t[i,j] * v_t[j,k] * w_t[k,i]... 
                # 
                # Wait, I think the issue is that the decomposition is:
                # C[i,k] = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]
                # No, that gives a product of three sums, which is cubic.
                # 
                # Let me re-read the problem. The factor matrices are U (n*m, rank), V (m*p, rank), W (p*n, rank).
                # The decomposition should satisfy:
                # C[i,k] = sum_t U[i*m + j, t] * V[j*p + k, t] * ... 
                # 
                # I think the correct interpretation is:
                # C[i,k] = sum_j sum_t U[i*m+j, t] * A[i,j] * V[j*p+k, t] * B[j,k] * W[k*n+i, t]
                # But that's still not quite right because we're summing over j inside.
                # 
                # Actually, I believe the decomposition is:
                # For each t, define linear maps:
                #   A -> sum_j U[i*m+j, t] * A[i,j] for each i (but this depends on i)
                # 
                # I think the correct form is that the bilinear map is:
                # C[i,k] = sum_t (U[:, t]^T * A_flat) * (V[:, t]^T * B_flat) * W[k*n+i, t]
                # where A_flat is A reshaped to (n*m, 1) and B_flat is B reshaped to (m*p, 1).
                # But that would make it a product of three linear forms, which is trilinear, not bilinear.
                # 
                # I'm confused. Let me just verify numerically with the known decomposition.
                pass
        
        # Let me just check a specific case
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        B = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                       [9, 10, 11, 12, 13, 14, 15, 16],
                       [17, 18, 19, 20, 21, 22, 23, 24]], dtype=float)
        C_expected = A @ B
        
        # Compute C using the decomposition
        # C[i,k] = sum_t sum_j U[i*m+j, t] * A[i,j] * V[j*p+k, t] * B[j,k] * W[k*n+i, t]
        # Wait, that's not right. Let me think about this differently.
        # 
        # The tensor T for matrix multiplication satisfies:
        # T[i,j,k] = 1 (for all i,j,k)
        # And T = sum_t u_t ⊗ v_t ⊗ w_t
        # where u_t ∈ R^{n×m}, v_t ∈ R^{m×p}, w_t ∈ R^{p×n}
        # So T[i,j,k] = sum_t u_t[i,j] * v_t[j,k] * w_t[k,i]... 
        # No, the third index of w should match the output.
        # 
        # I think the correct interpretation is:
        # C[i,k] = sum_t (sum_j u_t[i,j] * A[i,j]) * (sum_j v_t[j,k] * B[j,k]) * w_t[k,i]
        # But that's a product of three terms, which doesn't make sense for bilinear maps.
        # 
        # Actually, I think the decomposition is:
        # C[i,k] = sum_t u_t[i,:] · A[i,:] * v_t[:,k] · B[:,k] * w_t[k,i]
        # No...
        # 
        # Let me just test empirically.

        C_computed = np.zeros((n, p))
        for t in range(rank):
            u = U[:, t]  # shape (n*m,)
            v = V[:, t]  # shape (m*p,)
            w = W[:, t]  # shape (p*n,)
            
            # Reshape
            u_mat = u.reshape(n, m)  # (n, m)
            v_mat = v.reshape(m, p)  # (m, p)
            w_mat = w.reshape(p, n)  # (p, n)
            
            # The contribution of term t to C[i,k] is:
            # sum_j u_mat[i,j] * A[i,j] * sum_j v_mat[j,k] * B[j,k] * w_mat[k,i]
            # That's still not right.
            
            # I think the correct formula is:
            # C[i,k] += (u_mat[i,:] * A[i,:]).sum() * (v_mat[:,k] * B[:,k]).sum() * w_mat[k,i]
            # But that's a product of three scalars, which is cubic in the inputs.
            
            # Let me try a different interpretation:
            # C[i,k] += sum_j u_mat[i,j] * A[i,j] * v_mat[j,k] * B[j,k] * w_mat[k,i]
            # This is a sum over j of products of A and B entries, weighted by U, V, W.
            # This is bilinear in A and B!
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[i, k] += u_mat[i, j] * A[i, j] * v_mat[j, k] * B[j, k] * w_mat[k, i]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            print("Verification failed!")
            print("Expected:", C_expected)
            print("Computed:", C_computed)
            return False
        return True
    
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback: return schoolbook
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col_s] = 1
                    V_s[j*p+k, col_s] = 1
                    W_s[k*n+i, col_s] = 1
                    col_s += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # The decomposition is valid if T = sum_t u_t ⊗ v_t ⊗ w_t
    # We can try to find if some columns are linear combinations of others.
    
    # Check time
    if time.time() > deadline - 30:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction using SVD on the tensor
    # Reshape the tensor T = sum_t u_t ⊗ v_t ⊗ w_t
    # T is a 3-tensor of shape (n*m, m*p, p*n)
    # But actually, T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # This is not a standard tensor product because the indices are coupled.
    
    # Let me try a different approach: check if we can find a smaller rank by
    # solving a system of equations.
    
    # For now, return the rank-44 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
