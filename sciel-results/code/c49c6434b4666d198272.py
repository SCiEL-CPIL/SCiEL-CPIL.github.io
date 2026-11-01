"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
    # where 0<=i<n, 0<=j<m, 0<=k<p
    
    # We want U (n*m, r), V (m*p, r), W (p*n, r) such that
    # sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
    
    # Strategy: Randomized search for low rank decompositions
    # Start from rank 32 (half of 48) and try to find a solution.
    # If found, return it. If not, try slightly higher ranks or different random seeds.
    
    # The trivial rank is 48.
    # For 2x2x2, rank is 7.
    # For 2x2x4, rank is 11? Or lower?
    # For 2x2x12, let's guess rank might be around 30-35.
    
    # We will use a simple iterative method:
    # Initialize U, V, W randomly.
    # Iterate: 
    #   Fix V, W, solve for U (linear least squares)
    #   Fix U, W, solve for V
    #   Fix U, V, solve for W
    # This is similar to ALS (Alternating Least Squares) for CP decomposition.
    
    # However, ALS tends to get stuck in local minima.
    # Given the small size, we can try multiple restarts.
    
    # Let's try to find a decomposition with rank r.
    # We'll test r from 24 up to 47.
    
    # To speed up, we can represent T as a matrix unfolding.
    # T_1 (n*m x (m*p * p*n)) is too big? 4 x (24 * 24) = 4 x 576.
    # Actually, the CP rank decomposition condition is:
    # T_1 = U (V * W)  (Hadamard product of rows of V and W? No, that's not right for CP)
    
    # For CP decomposition:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # Unfolding T_1 (matrix of size I x (J*K)):
    # T_1 = U (W ⊙ V)  where ⊙ is Khatri-Rao product (column-wise Kronecker).
    # So we want T_1 ≈ U (W ⊙ V).
    
    # Let's implement a simple search.
    
    # Flatten T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T1 = T.reshape(n*m, m*p * p*n)
    
    best_rank = 48
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 24 to 47
    for r in range(24, 48):
        if time.time() > deadline - 5:
            break
            
        # Try a few random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize to avoid scaling issues
            # U = U / np.linalg.norm(U, axis=0, keepdims=True)
            
            # ALS iterations
            for it in range(20):
                # Solve for U: T1 = U (W ⊙ V)
                WV = np.kron(W.T, V.T).T # Shape (p*n*m*p, r) ? No.
                # Khatri-Rao product of W and V:
                # W is (p*n, r), V is (m*p, r)
                # W ⊙ V should be (p*n * m*p, r) where column r is kron(w_r, v_r)
                # np.kron(W, V) is not Khatri-Rao.
                # Khatri-Rao of A (I x R) and B (J x R) is (I*J x R)
                # Column r is A[:, r] ⊗ B[:, r]
                
                # We want to solve T1 = U * (W ⊙ V)
                # Let M = W ⊙ V
                M = np.zeros((p*n * m*p, r))
                for col in range(r):
                    M[:, col] = np.kron(W[:, col], V[:, col])
                
                # T1 is (n*m, m*p * p*n)
                # M is (m*p * p*n, r)
                # We need T1 = U M
                # This is a linear least squares problem for U.
                try:
                    U, _, _, _ = np.linalg.lstsq(M, T1.T, rcond=None)
                    U = U.T
                except:
                    break
                
                # Solve for V: T2 = V (U ⊙ W)
                T2 = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
                MW = np.zeros((n*m * p*n, r))
                for col in range(r):
                    MW[:, col] = np.kron(U[:, col], W[:, col])
                try:
                    V, _, _, _ = np.linalg.lstsq(MW, T2.T, rcond=None)
                    V = V.T
                except:
                    break
                
                # Solve for W: T3 = W (U ⊙ V)
                T3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
                MV = np.zeros((n*m * m*p, r))
                for col in range(r):
                    MV[:, col] = np.kron(U[:, col], V[:, col])
                try:
                    W, _, _, _ = np.linalg.lstsq(MV, T3.T, rcond=None)
                    W = W.T
                except:
                    break
                
                # Check error
                # Reconstruct T
                T_rec = np.zeros((n*m, m*p, p*n))
                for col in range(r):
                    T_rec += np.einsum('i,j,k->ijk', U[:, col], V[:, col], W[:, col])
                
                err = np.linalg.norm(T_rec - T)
                if err < 1e-5:
                    found = True
                    break
            
            if found:
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break # Found a valid decomposition for this rank, move to next lower rank? 
                       # Actually, we are iterating r from 24 to 47. 
                       # If we find one at r=30, we should try to find one at r=29? 
                       # The loop goes 24, 25, 26... So if we find one at 30, we have already tried 24-29.
                       # So we can break out of the restart loop, but we should continue to next r? 
                       # No, if we found one at r, we want the MINIMAL rank. 
                       # Since we start at 24, if we find one at 30, it means 24-29 failed.
                       # So 30 is the best so far. We can break the outer loop if we want to save time, 
                       # but maybe 29 was just unlucky? 
                       # Given time constraints, finding one at r is good. 
                       # Let's assume the first found is good enough or try a bit more?
                       # Let's just break and return this rank.
                break
    
    if best_U is not None:
        # Convert to lists
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to trivial rank 48
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
