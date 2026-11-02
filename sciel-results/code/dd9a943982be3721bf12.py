"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)  
    # W: (p*n, rank) = (12, rank)
    
    def schoolbook():
        rank = n * m * p  # 48
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is correct."""
        rank = U.shape[1]
        # Build the tensor T_{(i,k),(i',j'),(j',k')} = delta_{i,i'}*delta_{k,k'}
        # Reconstruction: for each output (i,k), sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r]*A[i',j']) * (sum_{j',k'} V[j'*p+k', r]*B[j',k'])
        # This should equal sum_j A[i,j]*B[j,k]
        
        # Test with random matrices
        rng = np.random.RandomState(42)
        for _ in range(3):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            
            # Compute via decomposition
            # L_A[r] = sum_{i',j'} U[i',j',r] * A[i',j']
            L_A = np.zeros(rank)
            for r in range(rank):
                L_A[r] = np.sum(U[:, r] * A.flatten())
            
            L_B = np.zeros(rank)
            for r in range(rank):
                L_B[r] = np.sum(V[:, r] * B.flatten())
            
            C_recon = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    C_recon[i, k] = np.sum(W[k*n + i, :] * L_A * L_B)
            
            if np.max(np.abs(C_recon - C_expected)) > tol:
                return False
        return True
    
    def try_reduce_rank(U, V, W, target_rank, time_limit):
        """Try to reduce rank to target_rank by zeroing out columns."""
        start_time = time.time()
        rank = U.shape[1]
        if target_rank >= rank:
            return U, V, W, rank
        
        # Try all subsets of size target_rank from the current rank
        # This is C(48, 24) which is huge, so we need a smarter approach
        # Instead, try to find a subset of columns that spans the tensor
        
        # Greedy: iteratively remove columns if the decomposition still works
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        current_rank = rank
        
        # Try removing columns one at a time
        removed = []
        for col in range(rank):
            if time.time() - start_time > time_limit:
                break
            if col in removed:
                continue
            # Try removing this column
            mask = [c for c in range(rank) if c != col and c not in removed]
            if len(mask) < target_rank:
                break
            
            U_test = U_cur[:, mask]
            V_test = V_cur[:, mask]
            W_test = W_cur[:, mask]
            
            if verify(U_test, V_test, W_test):
                removed.append(col)
                current_rank = len(mask)
        
        if current_rank < rank:
            mask = [c for c in range(rank) if c not in removed]
            return U_cur[:, mask], V_cur[:, mask], W_cur[:, mask], current_rank
        
        return U, V, W, rank
    
    # Start with schoolbook
    U, V, W = schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start_time = time.time()
    time_budget = 120  # seconds, with safety margin
    
    # Try to reduce rank
    # First, try greedy removal from full rank
    U_reduced, V_reduced, W_reduced, rank_reduced = try_reduce_rank(U, V, W, 1, time_budget * 0.3)
    
    if rank_reduced < best_rank:
        best_rank = rank_reduced
        best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
    
    # Try different random initial orderings to find better reductions
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5:
        rng = np.random.RandomState(123)
        for trial in range(5):
            if time.time() - start_time > time_budget:
                break
            # Randomly permute the columns
            perm = rng.permutation(best_rank)
            U_perm = best_U[:, perm]
            V_perm = best_V[:, perm]
            W_perm = best_W[:, perm]
            
            U_try, V_try, W_try, rank_try = try_reduce_rank(U_perm, V_perm, W_perm, 1, remaining_time / 6)
            
            if rank_try < best_rank:
                best_rank = rank_try
                best_U, best_V, best_W = U_try, V_try, W_try
    
    # Convert to lists for output
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
