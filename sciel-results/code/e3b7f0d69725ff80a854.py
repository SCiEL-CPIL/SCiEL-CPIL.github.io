"""Matrix multiplication 2x5x5 - attempt at lower rank via ALS and integer rounding."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Attempt to find a rank-minimal tensor decomposition for 2x5x5 matrix multiplication.
    Uses Alternating Least Squares (ALS) to find a low-rank approximation, 
    then attempts to round to integer/half-integer entries and verify exactness.
    Falls back to schoolbook rank 50 if no better solution is found within time limit.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Note: The problem statement says W[k*n+i, r]. 
    # So the third index of the tensor is k*n+i.
    # U index: i*m+j
    # V index: j*p+k
    # W index: k*n+i
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                target[idx_u, idx_v, idx_w] = 1.0

    def verify_solution(U, V, W):
        """Verify if U, V, W exactly reconstruct the target tensor with integer/half-integer entries."""
        # Check integer/half-integer constraint
        # Entries should be k/2 for integer k
        # Multiply by 2 and check if integer
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        # Check exact reconstruction
        # Reconstruct tensor
        # T_comp[a, b, c] = sum_d U[a, d] * V[b, d] * W[c, d]
        # Efficient computation using einsum
        T_comp = np.einsum('ad,bd,cd->abc', U, V, W)
        
        # Check if T_comp matches target exactly (within floating point tolerance for the sum, 
        # but since inputs are half-integers, the sum is a rational number with denominator dividing 8? 
        # No, U*V*W is (k1/2)*(k2/2)*(k3/2) = k1k2k3/8. Sum of these is integer/8.
        # Target is 1. So we need sum to be exactly 1.
        # Floating point error might be an issue. We should check with a tighter tolerance or 
        # use fractions if we were really strict, but for 120s, float64 is likely ok if we are close.
        
        if not np.allclose(T_comp, target, atol=1e-9):
            return False
            
        return True

    def round_to_half_integer(mat):
        """Round matrix entries to nearest half-integer."""
        return np.round(mat * 2) / 2.0

    def try_rank(rank, max_iterations=100, tol=1e-10):
        """
        Attempt to find a rank-r decomposition using ALS.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline:
            return None

        # Random initialization
        np.random.seed(42 + rank)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1

        prev_err = np.inf
        for it in range(max_iterations):
            if time.time() > deadline:
                return None

            # 1. Fix V, W, update U
            # T_comp[a, b, c] = sum_d U[a, d] V[b, d] W[c, d]
            # For fixed a, we want to minimize sum_{b,c} (T[a,b,c] - sum_d U[a,d] V[b,d] W[c,d])^2
            # This is a linear least squares problem for U[:, :].
            # Let M_U be a matrix where M_U[a, b*c + d] ... no, let's reshape.
            # T_reshaped: (n*m, m*p*p*n) ? No.
            # We can form a matrix A_U of shape (n*m, (m*p)*(p*n)) where
            # A_U[a, (b, c)] = sum_d V[b, d] W[c, d] ? No, that's not right.
            
            # The term is sum_d U[a, d] * (V[b, d] * W[c, d])
            # Let's construct the design matrix for U.
            # For each a, we have variables U[a, 0..rank-1].
            # The error is sum_{b,c} (T[a,b,c] - sum_d U[a,d] V[b,d] W[c,d])^2
            # This is equivalent to:
            # Let Y_a be the vector of T[a, :, :] flattened (size m*p*p*n).
            # Let X_a be a matrix of size (m*p*p*n, rank) where
            # X_a[(b, c), d] = V[b, d] * W[c, d]
            # Then we solve X_a @ U[a, :] = Y_a
            
            # Construct X (same for all a, structure is fixed by V, W)
            # V is (m*p, rank), W is (p*n, rank)
            # We need outer product of V and W along the rank dimension?
            # V[:, d] is vector of size m*p. W[:, d] is vector of size p*n.
            # We want a matrix of size (m*p * p*n, rank) where entry ((b, c), d) = V[b, d] * W[c, d]
            # This can be constructed as:
            # X = np.zeros((m*p * p*n, rank))
            # for d in range(rank):
            #     X[:, d] = (V[:, d] * W[:, d]).flatten() ? No.
            #     V[:, d] is length m*p. W[:, d] is length p*n.
            #     The outer product V[:, d] (x) W[:, d] is (m*p, p*n).
            #     Flatten it to (m*p*p*n,).
            
            # Efficient construction:
            # X_d = np.kron(V[:, d], W[:, d]) ? 
            # np.kron(A, B) for A (m,1), B (n,1) gives (mn, 1).
            # Yes.
            
            X = np.zeros((m*p * p*n, rank))
            for d in range(rank):
                X[:, d] = np.kron(V[:, d], W[:, d])
            
            # Y is target reshaped: (n*m, m*p*p*n)
            Y = target.reshape(n*m, -1)
            
            # Solve U = Y @ X^T (X X^T)^-1 ? Or use lstsq.
            # U[a, :] = lstsq(X, Y[a, :])
            U_new = np.zeros_like(U)
            for a in range(n*m):
                # Solve X @ u = Y[a, :]
                # X is (N, rank), Y[a] is (N,)
                # u is (rank,)
                sol, _, _, _ = np.linalg.lstsq(X, Y[a], rcond=None)
                U_new[a, :] = sol
            U = U_new

            # 2. Fix U, W, update V
            # Symmetric logic.
            # For each b, minimize sum_{a,c} (T[a,b,c] - sum_d U[a,d] V[b,d] W[c,d])^2
            # Variables: V[b, :]
            # Design matrix X_V of size (n*m * p*n, rank)
            # X_V[(a, c), d] = U[a, d] * W[c, d]
            X_V = np.zeros((n*m * p*n, rank))
            for d in range(rank):
                X_V[:, d] = np.kron(U[:, d], W[:, d])
            
            Y_V = target.transpose(0, 1, 2).reshape(m*p, -1) # Shape (m*p, n*m*p*n)
            # Wait, target is (n*m, m*p, p*n).
            # We want to group by b (index 1).
            # Y_V[b, (a, c)] = T[a, b, c]
            # Reshape target to (n*m, m*p, p*n).
            # Move axis 1 to front: (m*p, n*m, p*n).
            # Flatten: (m*p, n*m*p*n).
            Y_V = target.transpose(1, 0, 2).reshape(m*p, -1)
            
            V_new = np.zeros_like(V)
            for b in range(m*p):
                sol, _, _, _ = np.linalg.lstsq(X_V, Y_V[b], rcond=None)
                V_new[b, :] = sol
            V = V_new

            # 3. Fix U, V, update W
            # For each c, minimize sum_{a,b} (T[a,b,c] - sum_d U[a,d] V[b,d] W[c,d])^2
            # Variables: W[c, :]
            # Design matrix X_W of size (n*m * m*p, rank)
            # X_W[(a, b), d] = U[a, d] * V[b, d]
            X_W = np.zeros((n*m * m*p, rank))
            for d in range(rank):
                X_W[:, d] = np.kron(U[:, d], V[:, d])
            
            Y_W = target.transpose(2, 0, 1).reshape(p*n, -1)
            
            W_new = np.zeros_like(W)
            for c in range(p*n):
                sol, _, _, _ = np.linalg.lstsq(X_W, Y_W[c], rcond=None)
                W_new[c, :] = sol
            W = W_new

            # Check error
            T_comp = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.sum((T_comp - target)**2)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err

        # After ALS, check if we can round to half-integers
        # Try rounding
        U_round = round_to_half_integer(U)
        V_round = round_to_half_integer(V)
        W_round = round_to_half_integer(W)
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
        
        # Try a few local perturbations if close? 
        # For now, if not exact, fail.
        
        return None

    # Try ranks from 49 down to 40 (limited search)
    # If rank 50 is the minimum, this will fail and we return 50.
    # We only try a few ranks to stay within time budget.
    best_U, best_V, best_W = None, None, None
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    # Try to beat schoolbook
    # Start with 49, go down to 45 to save time
    for rank in range(49, 44, -1):
        if time.time() > deadline - 10: # Stop trying new ranks if time is low
            break
        res = try_rank(rank)
        if res is not None:
            best_U, best_V, best_W = res
            break
            
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
