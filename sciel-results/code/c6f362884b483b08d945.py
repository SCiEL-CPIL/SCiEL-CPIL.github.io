"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def to_dict(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W],
        }

    def verify(U, V, W, tol=1e-9):
        """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{i,k}... 
        Actually the tensor is T[i,j,k] = delta_{i,k} for matmul (n x m) * (m x p).
        Wait: (A * B)[i,k] = sum_j A[i,j]*B[j,k].
        The tensor for matrix multiplication is T[i,j,k] where the decomposition is:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
        For standard matmul, T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
        Actually the matrix multiplication tensor has entries T[i,j,k] = delta_{i,k}? No.
        
        Let me think again. The matrix multiplication tensor T is defined such that:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        In tensor form, we want:
        T[i,j,k] such that C = A*B means C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
        
        Actually, the standard tensor decomposition for matmul:
        We want U, V, W such that for all i,j,k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        Wait no. The tensor T_{ijk} for multiplying an (n x m) matrix by an (m x p) matrix
        is simply T_{ijk} = 1 for all valid i,j,k. Because:
        C[i,k] = sum_j A[i,j] B[j,k]
        
        So the "tensor" we're decomposing has all ones? That doesn't seem right for a rank problem.
        
        Let me re-read the task. "Matrix multiplication 4,4,9" - find rank-minimal exact tensor decomposition.
        
        The matrix multiplication tensor T is a 3-tensor of dimensions (n*m) x (m*p) x (p*n) where:
        T[a, b, c] with a = i*m+j, b = j'*p+k, c = k'*n+i'
        T[a,b,c] = 1 if j == j' and i == i' and k == k', else 0
        
        Wait, that's the standard definition. Let me verify with the schoolbook:
        U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for the same (i,j,k)
        Then sum_c U[i*m+j,c]*V[j'*p+k',c]*W[k''*n+i'',c] = 1 iff (i,j,k) = (i',j',k'') and i=i'', k=k''
        = 1 iff i=i', j=j'', k=k'', i=i'', k=k'' which is consistent.
        
        So T[i*m+j, j'*p+k', k''*n+i''] = 1 if j=j'' and i=i'' and k=k'', else 0.
        Simplifying: T[a,b,c] = 1 if the j-index of a equals the j-index of b, AND the i-index of c equals the i-index of a, AND the k-index of b equals the k-index of c.
        """
        # Build the target tensor
        # T[i*m+j, j2*p+k, k2*n+i2] = 1 if j==j2 and i==i2 and k==k2
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        rank = U_mat.shape[1]
        
        # Check a random sample of entries
        np.random.seed(42)
        for _ in range(2000):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            i2 = np.random.randint(n)
            j2 = np.random.randint(m)
            k2 = np.random.randint(p)
            
            a = i * m + j
            b = j2 * p + k
            c = k2 * n + i2
            
            # Target value
            target = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
            
            # Computed value
            val = np.dot(U_mat[a, :], V_mat[b, :]) * 0  # placeholder
            # Actually: sum_c U[a,c] * V[b,c] * W[c,c]... no
            # sum_c U[a,c] * V[b,c] * W[c_index, c] where c_index is the column index
            # The formula is: sum_{col} U[a,col] * V[b,col] * W[c_idx, col]
            # where c_idx = k2*n+i2
            val = sum(U_mat[a, col] * V_mat[b, col] * W_mat[c, col] for col in range(rank))
            
            if abs(val - target) > tol:
                return False
        return True

    # Try to find a lower rank decomposition using random initialization + gradient-free optimization
    # Strategy: Start with random rank-r matrices, use iterative refinement
    
    best_result = to_dict(U_sb, V_sb, W_sb)
    best_rank = rank_sb

    # Try decreasing rank
    for r in range(rank_sb - 1, 15, -1):  # Try ranks down to 16
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        np.random.seed(42)
        U_rand = np.random.randn(n * m, r) * 0.1
        V_rand = np.random.randn(m * p, r) * 0.1
        W_rand = np.random.randn(p * n, r) * 0.1
        
        # Use ALS (Alternating Least Squares) to fit the tensor
        # T[a,b,c] = sum_col U[a,col] V[b,col] W[c,col]
        
        # Precompute target tensor entries
        # T[a,b,c] = 1 if (a%m == b//p) and (a//m == c%n) and (b%p == c//n)
        
        max_iter = 50
        lr = 0.01
        
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            # Update U: for each a, U[a,:] should satisfy sum_col U[a,col] V[b,col] W[c,col] = T[a,b,c]
            # This is: for each (a,b,c), the constraint is linear in U[a,:]
            # Group by a: for fixed a, we have constraints for all (b,c)
            # U[a,:] * (V[b,:] ⊙ W[c,:]) = T[a,b,c]
            # This is a least squares problem
            
            # For efficiency, process in batches
            for a in range(n * m):
                # Collect all (b,c) pairs
                rhs = np.zeros(r)
                A_mat = np.zeros((m * p * p * n, r))  # too large, sample
                # Actually m*p = 36, p*n = 36, so 36*36 = 1296 constraints per a
                # That's manageable
                rows = []
                rhs_vec = []
                for b in range(m * p):
                    for c in range(p * n):
                        # Check if this (a,b,c) is a valid tensor entry
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        
                        target = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                        
                        # The row is V[b,:] ⊙ W[c,:]
                        row = V_rand[b, :] * W_rand[c, :]
                        rows.append(row)
                        rhs_vec.append(target)
                
                if rows:
                    A_arr = np.array(rows)
                    rhs_arr = np.array(rhs_vec)
                    # Solve least squares: A_arr @ U[a,:] = rhs_arr
                    try:
                        U_rand[a, :] = np.linalg.lstsq(A_arr, rhs_arr, rcond=None)[0]
                    except:
                        pass
            
            # Update V similarly
            for b in range(m * p):
                rows = []
                rhs_vec = []
                for a in range(n * m):
                    for c in range(p * n):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        
                        target = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                        
                        row = U_rand[a, :] * W_rand[c, :]
                        rows.append(row)
                        rhs_vec.append(target)
                
                if rows:
                    A_arr = np.array(rows)
                    rhs_arr = np.array(rhs_vec)
                    try:
                        V_rand[b, :] = np.linalg.lstsq(A_arr, rhs_arr, rcond=None)[0]
                    except:
                        pass
            
            # Update W similarly
            for c in range(p * n):
                rows = []
                rhs_vec = []
                for a in range(n * m):
                    for b in range(m * p):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        
                        target = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                        
                        row = U_rand[a, :] * V_rand[b, :]
                        rows.append(row)
                        rhs_vec.append(target)
                
                if rows:
                    A_arr = np.array(rows)
                    rhs_arr = np.array(rhs_vec)
                    try:
                        W_rand[c, :] = np.linalg.lstsq(A_arr, rhs_arr, rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if it % 5 == 0:
                # Quick check
                np.random.seed(123)
                ok = True
                for _ in range(100):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    i2 = np.random.randint(n)
                    j2 = np.random.randint(m)
                    k2 = np.random.randint(p)
                    a = i * m + j
                    b = j2 * p + k
                    c_idx = k2 * n + i2
                    target = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                    val = sum(U_rand[a, col] * V_rand[b, col] * W_rand[c_idx, col] for col in range(r))
                    if abs(val - target) > 0.1:
                        ok = False
                        break
                if ok:
                    # Full verification
                    if verify(U_rand, V_rand, W_rand, tol=1e-6):
                        best_result = to_dict(U_rand, V_rand, W_rand)
                        best_rank = r
                        break
        
        if time.time() > deadline - 30:
            break
    
    return best_result
# EVOLVE-BLOCK-END
