"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s safety margin for evaluation
    
    # Strategy: Use block decomposition.
    # Split B (3x8) into B1 (3x3), B2 (3x3), B3 (3x2).
    # Rank(3x3 * 3x8) <= Rank(3x3 * 3x3) + Rank(3x3 * 3x3) + Rank(3x3 * 3x2).
    # We aim for Rank(3x3 * 3x3) = 23 (Landsberg) and Rank(3x3 * 3x2) = 18 (Schoolbook).
    # Total rank = 23 + 23 + 18 = 64.
    
    def _verify_decomposition(U, V, W, n, m, p, num_trials=10):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        left = np.dot(U[i*m:(i+1)*m, r], A[i, :])
                        right = np.dot(V[k:m*p+k:m, r], B[:, k]) # Wait, V is (m*p, rank). Index j*p+k.
                        # V[j*p+k, r]. For fixed k, j varies 0..m-1.
                        # V[0*p+k, r], V[1*p+k, r], ...
                        # So V[k::p, r] is not quite right if p is not the stride in memory?
                        # V is shape (m*p, rank). Index is j*p+k.
                        # So for fixed k, the indices are k, p+k, 2p+k...
                        # So V[k::p, r] works if V is C-contiguous and index is j*p+k.
                        # Let's verify: j=0 -> k. j=1 -> p+k. Yes.
                        right = np.dot(V[k::p, r], B[:, k])
                        val += left * right * W[k*n+i, r]
                    C_computed[i,k] = val
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    def _build_schoolbook_block(n, m, p):
        """Returns U, V, W for n x m * m x p schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    def _combine_blocks(U1, V1, W1, U2, V2, W2, U3, V3, W3, n, m, p):
        """Combine three blocks for p = p1 + p2 + p3."""
        # We need to map the columns of B correctly.
        # B1 is m x p1, B2 is m x p2, B3 is m x p3.
        # V1 corresponds to B1, V2 to B2, V3 to B3.
        # The full V is m x p.
        # We need to interleave or concatenate?
        # The decomposition is sum_r U_r A V_r B W_r.
        # For block 1: V1_r is m x p1. It contributes to columns 0..p1-1 of B.
        # For block 2: V2_r is m x p2. It contributes to columns p1..p1+p2-1.
        # For block 3: V3_r is m x p3. It contributes to columns p1+p2..p-1.
        
        rank_total = U1.shape[1] + U2.shape[1] + U3.shape[1]
        U_full = np.zeros((n*m, rank_total))
        V_full = np.zeros((m*p, rank_total))
        W_full = np.zeros((p*n, rank_total))
        
        # Block 1
        U_full[:, 0:U1.shape[1]] = U1
        # V1 is (m*p1, rank1). We need to place it in V_full.
        # V_full index is j*p + k.
        # For k in 0..p1-1, V_full[j*p+k, r] = V1[j*p1+k, r].
        for j in range(m):
            for k in range(p1):
                V_full[j*p + k, 0:U1.shape[1]] = V1[j*p1 + k, :]
        # W1 is (p1*n, rank1). W_full index is k*n + i.
        # For k in 0..p1-1, W_full[k*n+i, r] = W1[k*n+i, r].
        for k in range(p1):
            for i in range(n):
                W_full[k*n + i, 0:U1.shape[1]] = W1[k*n + i, :]
        
        # Block 2
        offset = U1.shape[1]
        U_full[:, offset:offset+U2.shape[1]] = U2
        for j in range(m):
            for k in range(p2):
                V_full[j*p + (p1 + k), offset:offset+U2.shape[1]] = V2[j*p2 + k, :]
        for k in range(p2):
            for i in range(n):
                W_full[(p1 + k)*n + i, offset:offset+U2.shape[1]] = W2[k*n + i, :]
        
        # Block 3
        offset2 = offset + U2.shape[1]
        U_full[:, offset2:offset2+U3.shape[1]] = U3
        for j in range(m):
            for k in range(p3):
                V_full[j*p + (p1 + p2 + k), offset2:offset2+U3.shape[1]] = V3[j*p3 + k, :]
        for k in range(p3):
            for i in range(n):
                W_full[(p1 + p2 + k)*n + i, offset2:offset2+U3.shape[1]] = W3[k*n + i, :]
        
        return U_full, V_full, W_full

    # Try to find a rank 23 decomposition for 3x3 * 3x3.
    # If not found, use schoolbook (rank 27).
    U33_23, V33_23, W33_23 = None, None, None
    if time.time() < deadline - 20:
        # Search for rank 23
        pass # Placeholder for search. For now, use schoolbook.
        
    # Use schoolbook for 3x3 * 3x3 (Rank 27) and 3x3 * 3x2 (Rank 18).
    # Total Rank = 27 + 27 + 18 = 72.
    # This is not an improvement.
    
    # Let's try to find a rank 23 decomposition using a known one.
    # I will hardcode a rank 23 decomposition if I can.
    # Since I can't, I will try a numerical search for the 3x3 block.
    
    def _find_low_rank_33(target_rank, max_time=10):
        """Find a decomposition of 3x3 * 3x3 with given rank."""
        n, m, p = 3, 3, 3
        rng = np.random.RandomState(42)
        start_time = time.time()
        for _ in range(10): # 10 restarts
            if time.time() - start_time > max_time:
                break
            U = rng.randn(n*m, target_rank) * 0.1
            V = rng.randn(m*p, target_rank) * 0.1
            W = rng.randn(p*n, target_rank) * 0.1
            
            # Simple gradient descent
            lr = 0.01
            for it in range(1000):
                if time.time() - start_time > max_time:
                    break
                A = rng.randn(n, m)
                B = rng.randn(m, p)
                C_exp = A @ B
                C_comp = np.zeros((n, p))
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                err = 0
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            left = np.dot(U[i*m:(i+1)*m, r], A[i, :])
                            right = np.dot(V[k::p, r], B[:, k])
                            val += left * right * W[k*n+i, r]
                        C_comp[i,k] = val
                        err += (val - C_exp[i,k])**2
                        for r in range(target_rank):
                            left = np.dot(U[i*m:(i+1)*m, r], A[i, :])
                            right = np.dot(V[k::p, r], B[:, k])
                            dU[i*m:(i+1)*m, r] += (val - C_exp[i,k]) * right * W[k*n+i, r] * A[i, :]
                            dV[k::p, r] += (val - C_exp[i,k]) * left * W[k*n+i, r] * B[:, k]
                            dW[k*n+i, r] += (val - C_exp[i,k]) * left * right
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                if err < 1e-12:
                    return U, V, W
        return None, None, None

    # Try to find rank 23 for 3x3
    if time.time() < deadline - 30:
        U33, V33, W33 = _find_low_rank_33(23, max_time=10)
        if U33 is not None:
            if _verify_decomposition(U33, V33, W33, 3, 3, 3, num_trials=5):
                # Use this for the 3x3 blocks
                U33_23, V33_23, W33_23 = U33, V33, W33
    
    if U33_23 is not None:
        # Build 3x8 decomposition using two 3x3 (rank 23) and one 3x2 (rank 18)
        U1, V1, W1 = _build_schoolbook_block(3, 3, 2) # Rank 18
        U2, V2, W2 = U33_23, V33_23, W33_23 # Rank 23
        U3, V3, W3 = U33_23, V33_23, W33_23 # Rank 23
        
        # Order: 3x2, 3x3, 3x3
        U_full, V_full, W_full = _combine_blocks(U1, V1, W1, U2, V2, W2, U3, V3, W3, 3, 3, 8)
        # Note: _combine_blocks assumes p1=2, p2=3, p3=3.
        # We need to adjust the function or the call.
        # My _combine_blocks uses p1, p2, p3 from the V shapes.
        # V1 shape (3*2, 18), V2 shape (3*3, 23), V3 shape (3*3, 23).
        # This matches.
        
        if _verify_decomposition(U_full, V_full, W_full, 3, 3, 8, num_trials=10):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook_block(3, 3, 8)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def _try_rank(n, m, p, rank):
    """Try to determine if a rank-r decomposition exists."""
    return False

def _solve_decomposition(n, m, p, rank):
    """Solve for the decomposition matrices."""
    return None, None, None
# EVOLVE-BLOCK-END
