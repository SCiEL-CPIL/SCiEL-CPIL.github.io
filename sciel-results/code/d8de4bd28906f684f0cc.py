import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find a lower rank decomposition
    # We will try ranks from 25 to 39
    # We will use random search with integer entries in {-1, 0, 1}
    start_time = time.time()
    deadline = 120  # seconds
    
    for r in range(25, 40):
        if time.time() - start_time > deadline - 10:
            break
        # Try a few random initializations
        for _ in range(1000):
            if time.time() - start_time > deadline - 10:
                break
            U = np.random.choice([-1, 0, 1], size=(n * m, r))
            V = np.random.choice([-1, 0, 1], size=(m * p, r))
            # Solve for W using least squares
            # T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}
            # We can reshape U and V to form a matrix M where M_{(i,j), c} = U_{ic} V_{jc}
            # Then T_{(i,j), k} = M_{(i,j), c} W_{kc}
            # So W = M^+ T
            M = np.zeros((n * m * m, r)) # This is wrong, M should be (n*m, r) * (m*p, r) -> no
            # The tensor T has indices i, j, k.
            # We want to solve for W.
            # Fix U and V.
            # T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}
            # For each k, T_{i,j,k} is a matrix of size n x m.
            # T_k = sum_c (U_{:,c} V_{:,c}^T) W_{kc}
            # So T_k = sum_c W_{kc} (U_{:,c} V_{:,c}^T)
            # This is a linear combination of rank-1 matrices U_{:,c} V_{:,c}^T.
            # We can solve for W_{kc} using least squares.
            
            # Let's construct the matrix A where A_{(i,j), c} = U_{ic} V_{jc}
            A = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    for c in range(r):
                        A[i * m + j, c] = U[i * m + j, c] * V[j * p + 0, c] # Wait, V index is j*p+k? No, V is (m*p, r).
                        # V index for j is j*p + k? No, V is indexed by (j, k) in the product?
                        # The tensor is T_{i,j,k}.
                        # U is (n*m, r), V is (m*p, r), W is (p*n, r).
                        # T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
                        # This is not a simple matrix product.
                        # We need to solve for W given U and V.
                        # T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
                        # This is a system of linear equations for W.
                        # The number of equations is n*m*p = 40.
                        # The number of unknowns is p*n*r = 10*r.
                        # We can set up the matrix.
                        
            # Let's build the matrix for the linear system.
            # Unknowns: W_{k*n+i, c} for k in 0..p-1, i in 0..n-1, c in 0..r-1.
            # Equations: T_{i,j,k} for i in 0..n-1, j in 0..m-1, k in 0..p-1.
            # T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
            # This is not linear in W if U and V are fixed? Yes it is linear.
            # But the coefficient of W_{k'n+i', c'} in the equation for (i,j,k) is:
            # It is non-zero only if k'=k, i'=i, c'=c.
            # So the system is diagonal in the (k,i) blocks?
            # No, W_{k*n+i, c} appears in the equation for (i,j,k) with coefficient U_{i*m+j, c} V_{j*p+k, c}.
            # So for each (i,k), we have a system for W_{k*n+i, c} for c=0..r-1.
            # The equations for a fixed (i,k) are for j=0..m-1.
            # T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
            # This is a system of m equations for r unknowns W_{k*n+i, c}.
            # We can solve this using least squares.
            
            valid = True
            W = np.zeros((p * n, r))
            for i in range(n):
                for k in range(p):
                    # Build the matrix for this (i,k)
                    # Rows: j=0..m-1
                    # Cols: c=0..r-1
                    M_ik = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            M_ik[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    # RHS: T_{i,j,k} for j=0..m-1
                    # T_{i,j,k} is 1 if the product of A[i,j] and B[j,k] contributes to C[i,k]
                    # In matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k]
                    # So T_{i,j,k} is 1 for all i,j,k? No.
                    # The tensor T for matrix multiplication is T_{i,j,k} = 1 if the term A[i,j] B[j,k] is part of C[i,k].
                    # Actually, the tensor is T_{i,j,k} = 1 for all i,j,k?
                    # No, the tensor is defined such that C[i,k] = sum_{j} A[i,j] B[j,k].
                    # The decomposition is C[i,k] = sum_c (sum_j U_{i,j,c} A[i,j]) (sum_k V_{j,k,c} B[j,k]) W_{k,i,c}?
                    # No, the standard decomposition is:
                    # C[i,k] = sum_c W_{k,i,c} (sum_j U_{i,j,c} A[i,j]) (sum_j V_{j,k,c} B[j,k])?
                    # Let's check the indices.
                    # U is (n*m, r), V is (m*p, r), W is (p*n, r).
                    # The term is sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} A[i,j] B[j,k]?
                    # No, the tensor T_{i,j,k} multiplies A[i,j] B[j,k].
                    # So C[i,k] = sum_{j} A[i,j] B[j,k] = sum_{j} sum_c T_{i,j,k} U_{i,j,c} V_{j,k,c} W_{k,i,c} A[i,j] B[j,k]?
                    # This is not right.
                    # The decomposition is:
                    # C[i,k] = sum_c W_{k,i,c} (sum_j U_{i,j,c} A[i,j]) (sum_j V_{j,k,c} B[j,k])?
                    # No, that would be a product of two sums.
                    # The correct form is:
                    # C[i,k] = sum_c (sum_j U_{i,j,c} A[i,j]) (sum_j V_{j,k,c} B[j,k]) W_{k,i,c}?
                    # This is not a sum of rank-1 tensors.
                    # A rank-1 tensor is U_i V_j W_k.
                    # So C[i,k] = sum_c U_{i,c} V_{j,c} W_{k,c}?
                    # But C[i,k] depends on j.
                    # The tensor is T_{i,j,k}.
                    # C[i,k] = sum_{j} T_{i,j,k} A[i,j] B[j,k].
                    # We want T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
                    # So for fixed i,k, T_{i,j,k} = sum_c (U_{i,j,c} V_{j,k,c}) W_{k,i,c}.
                    # Let X_{j,c} = U_{i,j,c} V_{j,k,c}.
                    # Then T_{i,j,k} = sum_c X_{j,c} W_{k,i,c}.
                    # This is a linear system for W_{k,i,c}.
                    # The matrix is X (size m x r).
                    # The RHS is T_{i,j,k} for j=0..m-1.
                    # What is T_{i,j,k}?
                    # T_{i,j,k} is 1 if the term A[i,j] B[j,k] is included in C[i,k].
                    # In standard matrix multiplication, all terms are included.
                    # So T_{i,j,k} = 1 for all i,j,k.
                    # So we want X W = 1 (vector of ones).
                    # We can solve for W using least squares.
                    
                    # Build X
                    X = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            X[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    # RHS is ones
                    rhs = np.ones(m)
                    # Solve X W_ik = rhs
                    # W_ik is a vector of length r
                    try:
                        W_ik = np.linalg.lstsq(X, rhs, rcond=None)[0]
                        # Check if W_ik is close to integer/half-integer
                        # And check if the reconstruction is exact
                        # For now, we just store it
                        W[k * n + i, :] = W_ik
                    except:
                        valid = False
                        break
            if not valid:
                continue
            
            # Check if the decomposition is exact
            # T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
            # We need to check if this equals 1 for all i,j,k.
            # And also check if U, V, W have integer/half-integer entries.
            # We will check the reconstruction error.
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c] for c in range(r))
                        if abs(val - 1) > 1e-6:
                            error = 1
                            break
                    if error:
                        break
                if error:
                    break
            if error == 0:
                # Check if entries are integer/half-integer
                # We can round them and check again
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if np.allclose(U, U_rounded) and np.allclose(V, V_rounded) and np.allclose(W, W_rounded):
                    # Valid decomposition
                    best_rank = r
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    break
        if best_rank < rank_schoolbook:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
