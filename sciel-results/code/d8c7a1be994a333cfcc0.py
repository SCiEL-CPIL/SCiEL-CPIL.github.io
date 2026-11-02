import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 30 decomposition
    # 27 from 3x3x3 schoolbook
    # 1 from 3x3x1 (A1 b1)
    # 1 from 3x1x3 (a4 b2)
    # 1 from 3x1x1 (a4 b44)
    rank = 30
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: A1 (3x3) * B1 (3x3) -> C1 (3x3)
    # i in 0..2, j in 0..2, k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Part 2: A1 (3x3) * b1 (3x1) -> c1 (3x1)
    # b1 is column 3 (index 3) of B
    # i in 0..2, j in 0..2, k=3
    # T[i, j, 3] = 1
    # Rank 1: u_i=1, v_j=1, w_3=1
    for i in range(3):
        U[i*m + 0, col] += 1 # Wait, v_j is summed over j?
        # No, the term is sum_j A[i,j] b[j,3].
        # Tensor T[i,j,3] = 1.
        # Decomposition: sum_r u[i,r] v[j,r] w[3,r].
        # If we use rank 1, u[i,0]=1, v[j,0]=1, w[3,0]=1.
        # Then sum_r u[i,r] v[j,r] w[3,r] = 1*1*1 = 1.
        # But we need T[i,j,3] = 1 for all j.
        # So v[j,0] must be 1 for all j.
        # And u[i,0] = 1 for all i.
        # And w[3,0] = 1.
        pass
        
    # Let's do it properly.
    # We need to add 1 to U[i*m+j, col] for all j?
    # No, the decomposition is a sum of rank-1 tensors.
    # One rank-1 tensor is u ⊗ v ⊗ w.
    # u is vector of size n*m.
    # v is vector of size m*p.
    # w is vector of size p*n.
    
    # For Part 2:
    # We want T[i,j,3] = 1 for i in 0..2, j in 0..2.
    # Let r = col.
    # U[i*m + j, r] = 1 for all i, j in 0..2.
    # V[j*p + 3, r] = 1 for all j in 0..2.
    # W[3*n + i, r] = 1 for all i in 0..2.
    # Then sum_r U[i*m+j, r] V[j*p+3, r] W[3*n+i, r] = 1*1*1 = 1.
    # This works.
    
    for i in range(3):
        for j in range(3):
            U[i*m + j, col] = 1
            V[j*p + 3, col] = 1
        W[3*n + i, col] = 1
    col += 1
    
    # Part 3: a4 (3x1) * b2 (1x3) -> c2 (3x3)
    # a4 is column 3 (index 3) of A.
    # b2 is row 3 (index 3) of B, cols 0..2.
    # i in 0..2, j=3, k in 0..2.
    # T[i, 3, k] = 1.
    # Let r = col.
    # U[i*m + 3, r] = 1 for all i.
    # V[3*p + k, r] = 1 for all k in 0..2.
    # W[k*n + i, r] = 1 for all i, k.
    
    for i in range(3):
        U[i*m + 3, col] = 1
        for k in range(3):
            V[3*p + k, col] = 1
            W[k*n + i, col] = 1
    col += 1
    
    # Part 4: a4 (3x1) * b44 (1x1) -> c44 (3x1)
    # i in 0..2, j=3, k=3.
    # T[i, 3, 3] = 1.
    # Let r = col.
    # U[i*m + 3, r] = 1 for all i.
    # V[3*p + 3, r] = 1.
    # W[3*n + i, r] = 1 for all i.
    
    for i in range(3):
        U[i*m + 3, col] = 1
        V[3*p + 3, col] = 1
        W[3*n + i, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Verification
def verify():
    n, m, p = DIMENSIONS
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Random matrices
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    # Compute C from decomposition
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This should be 1 for all i,j,k? No.
    # The tensor T represents the multiplication.
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]?
    # No, the standard decomposition is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_k V[j*p+k, r] B[j,k]) W[k*n+i, r]?
    # No.
    # The decomposition is T = sum_r u_r ⊗ v_r ⊗ w_r.
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # And C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]?
    # No, T[i,j,k] is the coefficient.
    # For matrix multiplication, T[i,j,k] = 1.
    # So C[i,k] = sum_j 1 * A[i,j] B[j,k].
    # So we just need to check that T[i,j,k] = 1 for all i,j,k.
    
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # u ⊗ v ⊗ w
        # T[i,j,k] += u[i,j] v[j,k] w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                    
    # Check if T is all ones
    if not np.allclose(T, 1.0):
        print("Verification Failed")
        print(T)
    else:
        print("Verification Passed")

if __name__ == "__main__":
    verify()
