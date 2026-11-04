import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    timeout = 150.0 # Evaluator kills at 150s, keep margin
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, tol=1e-6):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] == 1
        r = U.shape[1]
        # Vectorized check
        # Create tensor of U, V, W products?
        # Loop is fine for small n,m,p
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_block_decomposition(n_b, m_b, p_b, rank_b, max_iter=2000):
        # Solve for rank_b decomposition of n_b x m_b * m_b x p_b
        # Returns U_b, V_b, W_b or None
        # n_b=3, m_b=3, p_b=2
        
        def objective(params):
            # params: U_flat (n_b*m_b*rank_b), V_flat (m_b*p_b*rank_b), W_flat (p_b*n_b*rank_b)
            size_U = n_b * m_b * rank_b
            size_V = m_b * p_b * rank_b
            size_W = p_b * n_b * rank_b
            
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U+size_V]
            W_flat = params[size_U+size_V:]
            
            U = U_flat.reshape(n_b, m_b, rank_b)
            V = V_flat.reshape(m_b, p_b, rank_b)
            W = W_flat.reshape(p_b, n_b, rank_b)
            
            # Compute residual
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Target is 1 for all i,j,k? No, the tensor T[i,j,k] = 1.
            # The decomposition is T[i,j,k] = sum_s U[i,j,s] V[j,k,s] W[k,i,s]
            # Wait, standard matrix mult tensor is T_{ijk} where i=row A, j=col A, k=row B (col C).
            # C_{ik} = sum_j A_{ij} B_{jk}.
            # So T_{ijk} = 1.
            # Factors: U_{ij,s}, V_{jk,s}, W_{ki,s}.
            # Note: W index is k (from B) and i (from A/C).
            
            # Calculate sum_s U[i,j,s] V[j,k,s] W[k,i,s]
            # We want this to be 1 for all i,j,k.
            
            # Efficient calculation:
            # For each s, we have matrices U_s (n_b x m_b), V_s (m_b x p_b), W_s (p_b x n_b).
            # The contribution to T is sum_s U_s * V_s * W_s? No.
            # T_{ijk} = sum_s U_{ij,s} V_{jk,s} W_{ki,s}.
            
            # Let's just loop or use einsum.
            # np.einsum('ijs,jks,ksi->ik', U, V, W) gives C_{ik}?
            # No, T_{ijk} is the tensor.
            # We want T_{ijk} = 1.
            
            # Residual for each i,j,k:
            # val = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
            
            # Vectorized:
            # U: (n_b, m_b, r)
            # V: (m_b, p_b, r)
            # W: (p_b, n_b, r)
            
            # We can compute the full tensor T_calc of shape (n_b, m_b, p_b)
            # T_calc[i,j,k] = sum_s U[i,j,s] V[j,k,s] W[k,i,s]
            
            # This is a bit complex for einsum directly without intermediate steps.
            # Let's do it in a loop over s for speed if rank is small, or use matrix mults.
            # Actually, for each s:
            # Term_s[i,j,k] = U[i,j,s] * V[j,k,s] * W[k,i,s]
            # This is an outer product of vectors?
            # U_s is n_b x m_b. V_s is m_b x p_b. W_s is p_b x n_b.
            # The product U_s[i,j] * V_s[j,k] * W_s[k,i] is the trace of (U_s * V_s * W_s)? No.
            # It's just element-wise product summed over s.
            
            # Let's compute T_calc using broadcasting.
            # U_expanded: (n_b, m_b, p_b, r) -- U[i,j,s] * 1
            # V_expanded: (n_b, m_b, p_b, r) -- 1 * V[j,k,s]
            # W_expanded: (n_b, m_b, p_b, r) -- 1 * 1 * W[k,i,s]
            
            # This creates large arrays. n_b=3, m_b=3, p_b=2, r=16.
            # 3*3*2*16 = 288. Very small.
            
            U_exp = U[:, :, np.newaxis, :] # (n_b, m_b, 1, r)
            V_exp = V[np.newaxis, :, :, :] # (1, m_b, p_b, r)
            W_exp = W[:, np.newaxis, np.newaxis, :] # (p_b, 1, 1, r) -- Wait, W is (p_b, n_b, r).
            # We need W[k,i,s]. So W_exp should be (p_b, n_b, 1, r) -> broadcast to (n_b, m_b, p_b, r)?
            # Indices: i (from U), j (from U,V), k (from V,W).
            # U[i,j,s] -> shape (n_b, m_b, 1, r)
            # V[j,k,s] -> shape (1, m_b, p_b, r)
            # W[k,i,s] -> W is (p_b, n_b, r). We need to align k with axis 2, i with axis 0.
            # W_exp = W.transpose(1, 0, 2) -> (n_b, p_b, r).
            # Then reshape to (n_b, 1, p_b, r).
            
            W_trans = W.transpose(1, 0, 2) # (n_b, p_b, r)
            W_exp = W_trans[:, np.newaxis, :, :] # (n_b, 1, p_b, r)
            
            T_calc = np.sum(U_exp * V_exp * W_exp, axis=3) # (n_b, m_b, p_b)
            
            target = np.ones((n_b, m_b, p_b))
            diff = T_calc - target
            return np.sum(diff**2)
        
        # Initial guess
        np.random.seed(42)
        size_U = n_b * m_b * rank_b
        size_V = m_b * p_b * rank_b
        size_W = p_b * n_b * rank_b
        
        # Random initialization
        params0 = np.random.randn(size_U + size_V + size_W) * 0.1
        
        # Optimize
        res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12})
        
        if res.success or res.fun < 1e-8:
            params = res.x
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U+size_V]
            W_flat = params[size_U+size_V:]
            
            U = U_flat.reshape(n_b, m_b, rank_b)
            V = V_flat.reshape(m_b, p_b, rank_b)
            W = W_flat.reshape(p_b, n_b, rank_b)
            
            # Try to round to half-integers
            # Check if rounding improves or maintains low error
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify rounded version
            U_exp = U_r[:, :, np.newaxis, :]
            V_exp = V_r[np.newaxis, :, :, :]
            W_trans = W_r.transpose(1, 0, 2)
            W_exp = W_trans[:, np.newaxis, :, :]
            T_calc = np.sum(U_exp * V_exp * W_exp, axis=3)
            err = np.max(np.abs(T_calc - 1.0))
            
            if err < 1e-6:
                return U_r, V_r, W_r
            else:
                # If rounding fails, maybe try rounding to integers?
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                U_exp = U_i[:, :, np.newaxis, :]
                V_exp = V_i[np.newaxis, :, :, :]
                W_trans = W_i.transpose(1, 0, 2)
                W_exp = W_trans[:, np.newaxis, :, :]
                T_calc = np.sum(U_exp * V_exp * W_exp, axis=3)
                err = np.max(np.abs(T_calc - 1.0))
                if err < 1e-6:
                    return U_i, V_i, W_i
                
                # If both fail, return None (or the unrounded if we accept floats? No, task requires integer/half-integer)
                return None
        return None

    # Try block decomposition for 3x3 x 3x2
    # Rank 16
    if time.time() - start_time < timeout - 20:
        U_b, V_b, W_b = try_block_decomposition(3, 3, 2, 16)
        if U_b is not None:
            # Construct full 3x3 x 3x8
            # n=3, m=3, p=8
            # 4 blocks
            rank_total = 4 * 16
            U_full = np.zeros((3*3, rank_total))
            V_full = np.zeros((3*8, rank_total))
            W_full = np.zeros((8*3, rank_total))
            
            for b in range(4):
                # Block b corresponds to columns 2*b to 2*b+1 of B and C
                # U is the same for all blocks?
                # In the tensor decomposition T[i,j,k] = sum_s U[i,j,s] V[j,k,s] W[k,i,s]
                # For block b, the indices are i, j, k' where k = 2*b + k'.
                # The auxiliary index s is local to the block, say s = 16*b + s'.
                
                # U_full[i, j, s] = U_b[i, j, s']
                # V_full[j, k, s] = V_b[j, k', s'] if k in block b, else 0
                # W_full[k, i, s] = W_b[k', i, s'] if k in block b, else 0
                
                # Fill U
                # U_b is (3,3,16). U_full is (9, 64).
                # U_full[i*3+j, 16*b + s'] = U_b[i, j, s']
                U_block = U_b.reshape(9, 16)
                U_full[:, 16*b:16*(b+1)] = U_block
                
                # Fill V
                # V_b is (3,2,16). V_full is (24, 64).
                # V_full[j*8+k, 16*b + s'] = V_b[j, k', s']
                # k = 2*b + k'
                V_block = V_b.reshape(6, 16)
                # V_full rows are j*8+k.
                # For a given b, the rows are j*8 + 2*b + k'.
                # We need to map V_block[j*2+k', s'] to V_full[j*8+2*b+k', 16*b+s']
                for j in range(3):
                    for k_prime in range(2):
                        row_full = j*8 + 2*b + k_prime
                        row_block = j*2 + k_prime
                        V_full[row_full, 16*b:16*(b+1)] = V_block[row_block, :]
                        
                # Fill W
                # W_b is (2,3,16). W_full is (24, 64).
                # W_full[k*3+i, 16*b + s'] = W_b[k', i, s']
                # k = 2*b + k'
                W_block = W_b.reshape(6, 16)
                for k_prime in range(2):
                    for i in range(3):
                        row_full = (2*b + k_prime)*3 + i
                        row_block = k_prime*3 + i
                        W_full[row_full, 16*b:16*(b+1)] = W_block[row_block, :]
                        
            # Verify full solution
            if verify(U_full, V_full, W_full):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                
    # If rank 16 failed, try rank 17
    if time.time() - start_time < timeout - 20:
        U_b, V_b, W_b = try_block_decomposition(3, 3, 2, 17)
        if U_b is not None:
            rank_total = 4 * 17
            U_full = np.zeros((3*3, rank_total))
            V_full = np.zeros((3*8, rank_total))
            W_full = np.zeros((8*3, rank_total))
            
            for b in range(4):
                U_block = U_b.reshape(9, 17)
                U_full[:, 17*b:17*(b+1)] = U_block
                
                V_block = V_b.reshape(6, 17)
                for j in range(3):
                    for k_prime in range(2):
                        row_full = j*8 + 2*b + k_prime
                        row_block = j*2 + k_prime
                        V_full[row_full, 17*b:17*(b+1)] = V_block[row_block, :]
                        
                W_block = W_b.reshape(6, 17)
                for k_prime in range(2):
                    for i in range(3):
                        row_full = (2*b + k_prime)*3 + i
                        row_block = k_prime*3 + i
                        W_full[row_full, 17*b:17*(b+1)] = W_block[row_block, :]
                        
            if verify(U_full, V_full, W_full):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback to schoolbook
    return schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
