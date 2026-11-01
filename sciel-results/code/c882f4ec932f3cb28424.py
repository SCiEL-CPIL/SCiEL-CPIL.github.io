"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Target tensor T of shape (U_size, V_size, W_size)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit

    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((U_size, rank))
        V = np.zeros((V_size, rank))
        W = np.zeros((W_size, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W):
        # U, V, W are numpy arrays of shape (size, rank)
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def objective(x):
        """
        Objective function for scipy.optimize.minimize.
        x is a flattened vector of U, V, W.
        """
        U_size_r = U_size * rank
        V_size_r = V_size * rank
        W_size_r = W_size * rank
        
        U_x = x[:U_size_r].reshape(U_size, rank)
        V_x = x[U_size_r:U_size_r+V_size_r].reshape(V_size, rank)
        W_x = x[U_size_r+V_size_r:].reshape(W_size, rank)
        
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U_x[:, c], V_x[:, c], W_x[:, c])
        
        return np.sum((T_calc - T)**2)

    def generate_initial_guesses(rank, num_guesses):
        """
        Generate initial guesses by merging columns of the schoolbook decomposition.
        Schoolbook rank is 18. We need to reduce it to 'rank'.
        We do this by summing pairs of columns.
        """
        schoolbook_rank = n * m * p
        U_sb = np.zeros((U_size, schoolbook_rank))
        V_sb = np.zeros((V_size, schoolbook_rank))
        W_sb = np.zeros((W_size, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        
        guesses = []
        if rank >= schoolbook_rank:
            return guesses
            
        # We need to merge (schoolbook_rank - rank) pairs of columns
        merges_needed = schoolbook_rank - rank
        
        # To create diverse guesses, we can randomly select which columns to merge.
        # A simple strategy: randomly partition the 18 columns into 'rank' groups.
        # Each group sums its columns.
        # This ensures the initial guess is "close" to schoolbook.
        
        for _ in range(num_guesses):
            # Randomly partition indices 0..17 into 'rank' non-empty subsets
            # This is a set partitioning problem.
            # Simple approach: assign each column to a random group 0..rank-1, ensuring no group is empty.
            # Then sum the columns in each group.
            
            # Ensure we have enough columns to fill 'rank' groups
            if rank > schoolbook_rank:
                continue
                
            # Random assignment of 18 columns to 'rank' bins
            # We need exactly 'rank' non-empty bins.
            # Let's pick 'rank' distinct columns to be "leaders", and assign the rest to random leaders.
            leaders = np.random.choice(schoolbook_rank, rank, replace=False)
            groups = {l: [l] for l in leaders}
            
            remaining = [i for i in range(schoolbook_rank) if i not in leaders]
            for idx in remaining:
                leader = np.random.choice(leaders)
                groups[leader].append(idx)
                
            # Construct U, V, W by summing columns in each group
            U_new = np.zeros((U_size, rank))
            V_new = np.zeros((V_size, rank))
            W_new = np.zeros((W_size, rank))
            
            for new_col_idx, leader in enumerate(leaders):
                cols = groups[leader]
                U_new[:, new_col_idx] = np.sum(U_sb[:, cols], axis=1)
                V_new[:, new_col_idx] = np.sum(V_sb[:, cols], axis=1)
                W_new[:, new_col_idx] = np.sum(W_sb[:, cols], axis=1)
                
            # Add small random noise to break symmetry
            U_new += np.random.randn(U_size, rank) * 0.1
            V_new += np.random.randn(V_size, rank) * 0.1
            W_new += np.random.randn(W_size, rank) * 0.1
            
            x0 = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
            guesses.append(x0)
            
        return guesses

    # Try ranks from 14 upwards
    # Theoretical lower bound is around 13-14. Schoolbook is 18.
    for rank in range(14, 19):
        if time.time() > deadline:
            break
            
        # Number of restarts
        max_restarts = 30 if rank < 17 else 10
        
        guesses = generate_initial_guesses(rank, max_restarts)
        
        for x0 in guesses:
            if time.time() > deadline:
                break
                
            try:
                # Use L-BFGS-B for large scale, but our problem is small (6*14 + 9*14 + 6*14 = 322 variables for rank 14)
                # TNC or L-BFGS-B are good.
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.success or res.fun < 1e-8:
                    x_opt = res.x
                    U_opt = x_opt[:U_size*rank].reshape(U_size, rank)
                    V_opt = x_opt[U_size*rank:U_size*rank+V_size*rank].reshape(V_size, rank)
                    W_opt = x_opt[U_size*rank+V_size*rank:].reshape(W_size, rank)
                    
                    # Snap and verify
                    result = snap_and_verify(U_opt, V_opt, W_opt)
                    if result:
                        U_final, V_final, W_final = result
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                        
            except Exception:
                continue

    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
