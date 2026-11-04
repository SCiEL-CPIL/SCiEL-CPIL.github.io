"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 138.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build the schoolbook decomposition as a fallback
    U_school = np.zeros((n * m, schoolbook_rank), dtype=float)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=float)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()

    def verify(U_mat, V_mat, W_mat, rank):
        """
        Check if U, V, W form a valid rank-r decomposition.
        Condition: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        """
        # Reshape for vectorized contraction
        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Using einsum: 'ijr,jkr,krk->ijk' doesn't work directly for W[k,n+i]
        # W is indexed by [k*n+i]. So W_reshaped is (p, n, rank) -> W[k, i, r]
        # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # Let's compute the tensor T_approx
        # T_approx[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        
        # We can do this in steps:
        # Step 1: Compute M[i,j,k,r] = U[i,j,r] * V[j,k,r]
        # This is a 4D tensor (n, m, p, rank)
        # U is (n, m, rank), V is (m, p, rank)
        # We need to align j.
        # M[i,j,k,r] = U[i,j,r] * V[j,k,r]
        
        # Efficient way:
        # Contract U and V over j? No, j is a free index in the result T[i,j,k].
        # Wait, the definition is C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor T[i,j,k] is 1 for all i,j,k.
        # The decomposition is T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        
        # Let's check the condition:
        # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        # Vectorized check:
        # U_mat: (n*m, rank)
        # V_mat: (m*p, rank)
        # W_mat: (p*n, rank)
        
        # We can iterate over r or use einsum.
        # Let's try to compute the full tensor T_approx of shape (n, m, p)
        
        # U_flat[i*m+j, r]
        # V_flat[j*p+k, r]
        # W_flat[k*n+i, r]
        
        # T_approx[i,j,k] = sum_r U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
        
        # This is a 3-mode tensor contraction.
        # We can compute it by iterating over r (rank is small, ~30)
        # Or using einsum if we reshape.
        
        # Reshape U to (n, m, rank)
        U_r = U_mat.reshape(n, m, rank)
        # Reshape V to (m, p, rank)
        V_r = V_mat.reshape(m, p, rank)
        # Reshape W to (p, n, rank)
        W_r = W_mat.reshape(p, n, rank)
        
        # We want T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        
        # Let's compute sum_r U_r[i,j,r] * V_r[j,k,r] first.
        # This gives a tensor A[i,j,k] of shape (n, m, p)
        # A[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r]
        
        # Using einsum: 'ijr,jkr->ijk'
        A = np.einsum('ijr,jkr->ijk', U_r, V_r)
        
        # Now T[i,j,k] = sum_k? No, k is free in T.
        # T[i,j,k] = A[i,j,k] * W_r[k,i,r] ? No, we already summed over r in A.
        # We need to sum over r again with W.
        # T[i,j,k] = sum_r (U_r[i,j,r] * V_r[j,k,r]) * W_r[k,i,r]
        # This is not A[i,j,k] * W... because W depends on r.
        
        # So we must keep r in the intermediate or do the full contraction.
        # Full contraction:
        # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        
        # We can do this in two steps:
        # 1. B[i,j,k,r] = U_r[i,j,r] * V_r[j,k,r]  (Shape: n, m, p, rank)
        # 2. T[i,j,k] = sum_r B[i,j,k,r] * W_r[k,i,r]
        
        # Step 1:
        # U_r is (n, m, rank), V_r is (m, p, rank)
        # We need to broadcast U and V to (n, m, p, rank)?
        # U[i,j,r] * V[j,k,r]
        # U_expanded = U_r[:, :, None, :] # (n, m, 1, rank)
        # V_expanded = V_r[None, :, :, :] # (1, m, p, rank)
        # B = U_expanded * V_expanded # (n, m, p, rank)
        
        U_exp = U_r[:, :, None, :]
        V_exp = V_r[None, :, :, :]
        B = U_exp * V_exp
        
        # Step 2:
        # T[i,j,k] = sum_r B[i,j,k,r] * W_r[k,i,r]
        # W_r is (p, n, rank) -> W_r[k, i, r]
        # We need to align indices: B is (n, m, p, rank), W is (p, n, rank)
        # We want to contract over r and match i, k.
        # T[i,j,k] = sum_r B[i,j,k,r] * W_r[k,i,r]
        
        # Reshape W to match axes for einsum?
        # W_r[k, i, r] -> indices are k, i, r.
        # B[i, j, k, r] -> indices are i, j, k, r.
        # Result T[i, j, k]
        
        # einsum: 'ijk r, k i r -> i j k'
        # Note: W_r is (p, n, rank) which corresponds to (k, i, r).
        # So W_r[k,i,r] is correct.
        
        T_approx = np.einsum('ijk r, k i r -> i j k', B, W_r)
        
        # Check if T_approx is all 1s
        if not np.allclose(T_approx, 1.0, atol=1e-8):
            return False
        return True

    def optimize_rank(r, max_iters=2000, lr=0.01):
        """
        Try to find a rank-r decomposition using vectorized gradient descent.
        """
        if time.time() > deadline - 5:
            return None

        np.random.seed(42 + r)
        
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Reshape
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            
            # Compute T_approx
            U_exp = U_r[:, :, None, :]
            V_exp = V_r[None, :, :, :]
            B = U_exp * V_exp # (n, m, p, r)
            T_approx = np.einsum('ijk r, k i r -> i j k', B, W_r)
            
            # Error
            E = T_approx - 1.0
            err = np.sum(E * E)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-10:
                break
            
            # Compute Gradients
            # dL/dU[i,j,r] = sum_k E[i,j,k] * V[j,k,r] * W[k,i,r]
            # dL/dV[j,k,r] = sum_i E[i,j,k] * U[i,j,r] * W[k,i,r]
            # dL/dW[k,i,r] = sum_j E[i,j,k] * U[i,j,r] * V[j,k,r]
            
            # Let's compute these using einsum
            # E is (n, m, p)
            # V_r is (m, p, r)
            # W_r is (p, n, r)
            
            # Grad_U[i,j,r] = sum_k E[i,j,k] * V[j,k,r] * W[k,i,r]
            # We can compute this as:
            # First, combine V and W?
            # Let M[j,k,i,r] = V[j,k,r] * W[k,i,r]
            # Then Grad_U[i,j,r] = sum_k E[i,j,k] * M[j,k,i,r]
            
            # This is getting complex to vectorize without huge intermediate arrays.
            # Let's do it step by step.
            
            # 1. Compute T_approx components needed for gradients
            # We already have B[i,j,k,r] = U[i,j,r] * V[j,k,r]
            # And T_approx[i,j,k] = sum_r B[i,j,k,r] * W[k,i,r]
            
            # Grad_U[i,j,r] = sum_k E[i,j,k] * V[j,k,r] * W[k,i,r]
            # Let's compute A1[i,j,k,r] = V[j,k,r] * W[k,i,r]
            # V_r[j,k,r] -> (m, p, r)
            # W_r[k,i,r] -> (p, n, r)
            # We need to align k.
            # A1[i,j,k,r] = V_r[j,k,r] * W_r[k,i,r]
            # V_exp2 = V_r[None, :, :, :] # (1, m, p, r)
            # W_exp2 = W_r[:, None, :, :] # (p, 1, n, r)
            # A1 = V_exp2 * W_exp2 # (1, m, p, n, r) -> broadcast to (n, m, p, n, r)?
            # Indices: i (from W), j (from V), k (common), i (from W again), r.
            # W_r[k, i, r] has i as the second index.
            # So W_exp2 should be (p, n, r) -> (p, 1, n, r) ? No.
            # We want A1[i, j, k, r] = V[j, k, r] * W[k, i, r]
            # V is (m, p, r). W is (p, n, r).
            # V[j,k,r] * W[k,i,r]
            # Let's use einsum: 'jkr, kir -> jkir'
            # This gives shape (m, p, n, r).
            # Let's call this M[j, k, i, r]
            
            M = np.einsum('jkr, kir -> jkir', V_r, W_r)
            # M[j, k, i, r]
            
            # Now Grad_U[i, j, r] = sum_k E[i, j, k] * M[j, k, i, r]
            # E is (n, m, p). M is (m, p, n, r).
            # We need to sum over k (index 2 in E, index 1 in M).
            # Result should be (n, m, r).
            # einsum: 'ijk, jkir -> ijr'
            # E[i,j,k] * M[j,k,i,r] -> sum k
            # Note: M's indices are j, k, i, r.
            # So we match E's k with M's k.
            # E[i,j,k] * M[j,k,i,r]
            # The 'i' in E matches the 'i' in M (3rd index of M).
            # The 'j' in E matches the 'j' in M (1st index of M).
            
            Grad_U = np.einsum('ijk, jkir -> ijr', E, M)
            
            # Grad_V[j, k, r] = sum_i E[i, j, k] * U[i, j, r] * W[k, i, r]
            # Let's compute N[i, j, k, r] = U[i, j, r] * W[k, i, r]
            # U_r[i, j, r] -> (n, m, r)
            # W_r[k, i, r] -> (p, n, r)
            # N[i, j, k, r] = U[i, j, r] * W[k, i, r]
            # einsum: 'ijr, kir -> ikjr' ? No.
            # We want N indexed by i, j, k, r.
            # U[i,j,r] * W[k,i,r]
            # U is (n, m, r). W is (p, n, r).
            # Let's do: 'ijr, kir -> ikjr'
            # i from U, j from U, r from U.
            # k from W, i from W, r from W.
            # Match i and r.
            # Result: i, j, k, r.
            
            N = np.einsum('ijr, kir -> ikjr', U_r, W_r)
            # N[i, j, k, r]
            
            # Grad_V[j, k, r] = sum_i E[i, j, k] * N[i, j, k, r]
            # E is (n, m, p). N is (n, m, p, r).
            # einsum: 'ijk, ikjr -> jkr'
            
            Grad_V = np.einsum('ijk, ikjr -> jkr', E, N)
            
            # Grad_W[k, i, r] = sum_j E[i, j, k] * U[i, j, r] * V[j, k, r]
            # Let's compute P[i, j, k, r] = U[i, j, r] * V[j, k, r]
            # This is exactly B[i, j, k, r] computed earlier!
            # B[i, j, k, r]
            
            # Grad_W[k, i, r] = sum_j E[i, j, k] * B[i, j, k, r]
            # E is (n, m, p). B is (n, m, p, r).
            # We want result indexed by k, i, r.
            # einsum: 'ijk, ijk r -> kir'
            # Match i, j, k.
            
            Grad_W = np.einsum('ijk, ijk r -> kir', E, B)
            
            # Update
            U -= lr * Grad_U.reshape(n * m, r)
            V -= lr * Grad_V.reshape(m * p, r)
            W -= lr * Grad_W.reshape(p * n, r)
            
            # Adaptive learning rate
            if err < best_err * 0.9:
                lr *= 1.1
            else:
                lr *= 0.9
                
            if lr < 1e-6:
                break

        # Final check
        if verify(best_U, best_V, best_W, r):
            return best_U, best_V, best_W
        return None

    # Try ranks from 35 down to 25
    # Schoolbook is 36.
    for r in range(35, 24, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Slightly different seed for each restart
            np.random.seed(100 + r * 10 + restart)
            
            res = optimize_rank(r, max_iters=1000, lr=0.05)
            if res is not None:
                U_r, V_r, W_r = res
                if verify(U_r, V_r, W_r, r):
                    best_rank = r
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
        
        if found:
            break

    # If no lower rank found, return schoolbook
    if best_rank == schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Convert to exact fractions if possible
    def to_exact(mat):
        result = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                rounded = np.round(v * 2) / 2
                if abs(v - rounded) < 1e-6:
                    result[i, j] = rounded
                else:
                    result[i, j] = v
        return result

    best_U = to_exact(best_U)
    best_V = to_exact(best_V)
    best_W = to_exact(best_W)

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
