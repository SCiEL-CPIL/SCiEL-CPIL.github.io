import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0

    def get_strassen_4x4x8():
        # ... (existing code) ...
        pass

    def get_laderman_4x4x8():
        # ... (Laderman code) ...
        pass

    # Try Laderman first
    try:
        U, V, W = get_laderman_4x4x8()
        # Verify
        n, m, p = 4, 4, 8
        rank = U.shape[1]
        # Random test
        np.random.seed(42)
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i, k] = sum_t U[i, t] * V[j, t] ... wait
        # The decomposition is C = sum_t (U[:,t] outer V[:,t]) outer W[:,t]? No.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # Tensor T_{ijk}.
        # Decomposition: T_{ijk} = sum_t U_{ij,t} V_{jk,t} W_{ki,t}.
        # So C_{ik} = sum_t W_{ki,t} * (sum_j U_{ij,t} A_{ij} * V_{jk,t} B_{jk})? No.
        # The formula is:
        # S_t = (sum_{i,j} U_{ij,t} A_{ij}) * (sum_{j,k} V_{jk,t} B_{jk})
        # C_{ik} = sum_t W_{ik,t} S_t
        
        S = np.zeros(rank)
        for t in range(rank):
            u = np.zeros((n, m))
            v = np.zeros((m, p))
            # Unflatten U[:,t] to n x m
            u = U[:,t].reshape(n, m)
            # Unflatten V[:,t] to m x p? 
            # Wait, for 4x4x4, V is 16x19. 
            # For 4x4x8, V is 32x38.
            # V indices: j*8+k.
            # In the sub-problem, V is 16x19.
            # My get_laderman_4x4x8 should return V of shape (32, 38).
            
            # Let's check the shape logic in get_laderman_4x4x8.
            # U: (16, 38)
            # V: (32, 38)
            # W: (32, 38)
            
            # For verification, I need to handle the flattening correctly.
            # U[:,t] is 16. Reshape to 4x4.
            # V[:,t] is 32. Reshape to 4x8.
            # W[:,t] is 32. Reshape to 8x4? Or 4x8?
            # W index is k*4+i. So reshape to 8x4? No, k is 0..7, i is 0..3.
            # W[k*4+i] -> W[k, i]. So reshape to (8, 4).
            
            u_mat = U[:,t].reshape(n, m)
            v_mat = V[:,t].reshape(m, p)
            w_mat = W[:,t].reshape(p, n)
            
            s_t = np.sum(u_mat * A) * np.sum(v_mat * B)
            S[t] = s_t
            
        C_computed = np.zeros((n, p))
        for t in range(rank):
            w_mat = W[:,t].reshape(p, n)
            C_computed += W[:,t].reshape(p, n).T * S[t] # W is p x n? 
            # C_{ik} = sum_t W_{ik,t} S_t.
            # W index k*4+i. So W[:,t] is length 32.
            # Reshape to (8, 4) gives W[k, i].
            # C[i, k] = sum_t W[k, i, t] S_t.
            # So C_computed[i, k] = sum_t W[k, i, t] S_t.
            # My loop: C_computed += w_mat.T * S[t] ?
            # w_mat is (8, 4). w_mat.T is (4, 8).
            # C_computed is (4, 8).
            # So C_computed[i, k] += w_mat[k, i] * S[t].
            # Yes.
            
        if not np.allclose(C_computed, C_expected):
            # Fallback to Strassen
            U, V, W = get_strassen_4x4x8()
        else:
            # Success
            pass
            
    except Exception as e:
        # Fallback to Strassen
        U, V, W = get_strassen_4x4x8()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
