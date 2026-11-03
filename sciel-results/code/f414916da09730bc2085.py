"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Leave 20s safety margin for 120s limit

    def schoolbook_solution():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return rank, U, V, W

    # Start with schoolbook rank
    best_rank, best_U, best_V, best_W = schoolbook_solution()

    def verify_decomposition(rank, U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that the tensor decomposition exactly represents matrix multiplication.
        
        The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        with T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product).
        
        In terms of matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        The tensor is T_{i,j,k} = 1 (coefficient of A_{i,j} B_{j,k} in C_{i,k})
        
        For decomposition: C_{i,k} = sum_r W_{k,i} * sum_r (sum_j U_{j,i} A_{i,j}) * (sum_j V_{j,k} B_{j,k})
        
        Actually, the standard tensor decomposition for matrix multiplication is:
        T_{i,j,k} = sum_r U_{i,j}^r * V_{j,k}^r * W_{k,i}^r
        
        where:
        - U has shape (n*m, rank), indexed by (i,j) -> row i*m+j
        - V has shape (m*p, rank), indexed by (j,k) -> row j*p+k  
        - W has shape (p*n, rank), indexed by (k,i) -> row k*n+i
        
        The decomposition is valid if for all (i,j,k):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        """
        # Check a sample of tensor entries to verify correctness
        # For efficiency, check all entries if rank is small, otherwise sample
        total_entries = n * m * p
        if total_entries * rank < 1e6:
            indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        else:
            rng = np.random.RandomState(42)
            indices = [(rng.randint(n), rng.randint(m), rng.randint(p)) for _ in range(1000)]
        
        for (i, j, k) in indices:
            u_row = U_mat[i * m + j, :]
            v_row = V_mat[j * p + k, :]
            w_row = W_mat[k * n + i, :]
            val = np.dot(u_row * v_row, w_row)
            if abs(val - 1.0) > tol:
                return False
        return True

    def try_random_search(start_rank, end_rank, num_trials=50000):
        """Try random rank-r decompositions and verify."""
        nonlocal best_rank, best_U, best_V, best_W
        rng = np.random.RandomState(12345)
        
        for rank in range(start_rank, end_rank + 1):
            if time.time() > deadline:
                break
            
            for trial in range(num_trials):
                if time.time() > deadline:
                    break
                
                # Generate random matrices with small integer/half-integer entries
                # Use entries from {-1, 0, 1} or {-0.5, 0, 0.5, 1, 1.5}
                scale = rng.choice([1, 2])
                U_mat = rng.choice([-1, 0, 1], size=(n*m, rank)) * (1.0 / scale)
                V_mat = rng.choice([-1, 0, 1], size=(m*p, rank)) * (1.0 / scale)
                W_mat = rng.choice([-1, 0, 1], size=(p*n, rank)) * (1.0 / scale)
                
                if verify_decomposition(rank, U_mat, V_mat, W_mat):
                    best_rank = rank
                    best_U = U_mat.copy()
                    best_V = V_mat.copy()
                    best_W = W_mat.copy()
                    return True
        
        return False

    # Try structured approaches based on known techniques
    
    # Approach: Use the fact that for rectangular matrices, we can sometimes
    # use block decomposition to reduce rank.
    
    # For 4x5 * 5x5, try to exploit structure
    # The bilinear complexity is at most n*m*p = 100
    
    # Try to find a better decomposition using algebraic techniques
    # One approach: use the "baby-step giant-step" style decomposition
    
    # Try random search for ranks below schoolbook
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25
    
    # Try structured ansatz: split the 5x5 into blocks
    # 5 = 3 + 2, 4 = 2 + 2
    
    # For now, try random search for a few ranks below 100
    # This is unlikely to find anything quickly, but let's try
    
    # Try rank 99, 98, ... down to 50 with limited trials
    for target_rank in [99, 98, 97, 96, 95]:
        if time.time() > deadline:
            break
        if try_random_search(target_rank, target_rank, num_trials=1000):
            break

    # If we haven't found anything better, return the schoolbook solution
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
