import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = get_schoolbook()
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try to find a lower rank decomposition
    # We will search for ranks from 80 down to 60
    for target_rank in range(80, 59, -5):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random small integers
        np.random.seed(target_rank)
        U = np.random.randint(-1, 2, size=(n * m, target_rank))
        V = np.random.randint(-1, 2, size=(m * p, target_rank))
        W = np.random.randint(-1, 2, size=(p * n, target_rank))
        
        # Function to compute error
        def compute_error(U, V, W):
            # E[i', j', j'', k'', i, k] = sum_c W[k, i, c] U[i', j', c] V[j'', k'', c] - delta
            # We can compute this using einsum
            # Term = np.einsum('ic, jc, kc -> ijk', U, V, W) ? No.
            # U is (n*m, R), V is (m*p, R), W is (p*n, R)
            # We need to reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            # Compute sum_c U[i', j', c] V[j'', k'', c] W[k'', i', c]?
            # The condition is:
            # sum_c W[k, i, c] U[i', j', c] V[j'', k'', c] = delta[i', i] delta[j', j''] delta[k'', k]
            # Let's compute the LHS for all indices
            # LHS[i', j', j'', k'', i, k] = sum_c U[i', j', c] V[j'', k'', c] W[k, i, c]
            # Note: W index is (k, i) in flattened form k*n+i.
            # So W_r[k, i, c].
            
            # Use einsum: 'ijc, jkc, kim -> ijkim'?
            # Indices:
            # U: i, j, c
            # V: j, k, c  (j'' is j, k'' is k)
            # W: k, i, c  (k is k'', i is i')? No.
            # W index in flattened is k*n+i. So W_r[k, i, c].
            # The term is U[i', j', c] * V[j'', k'', c] * W[k'', i', c]?
            # Wait, the seed code says:
            # u_c = sum U[i,j,c] A[i,j]
            # v_c = sum V[j,k,c] B[j,k]
            # C[i,k] = sum W[k,i,c] u_c v_c
            # Coeff of A[i',j'] B[j'',k''] in C[i,k] is:
            # sum_c W[k,i,c] U[i',j',c] V[j'',k'',c]
            # We want this to be 1 if i'=i, j'=j'', k''=k.
            
            # So we compute T[i', j', j'', k'', i, k] = sum_c U[i', j', c] V[j'', k'', c] W[k, i, c]
            # Note: W index is (k, i).
            
            # Let's use einsum
            # U: (n, m, R) -> indices i, j, c
            # V: (m, p, R) -> indices j, k, c
            # W: (p, n, R) -> indices k, i, c
            
            # We want to sum over c.
            # Result shape: (n, m, m, p, n, p)
            # T[i, j, j2, k2, i2, k2_out] ?
            # Let's map carefully.
            # i' -> i
            # j' -> j
            # j'' -> j2
            # k'' -> k2
            # i (output) -> i_out
            # k (output) -> k_out
            
            # Term: U[i, j, c] * V[j2, k2, c] * W[k2, i_out, c]
            # Note: W is indexed by (k, i_out) in the formula W[k, i, c].
            # So W_r[k2, i_out, c].
            
            T = np.einsum('ijc,jkc,kic->ijkik', U_r, V_r, W_r)
            # Wait, W_r is (p, n, R). Indices are k, i.
            # So W_r[k2, i_out, c].
            # The einsum 'ijc, jkc, kic' -> 'ijkik'
            # i: i'
            # j: j'
            # k: k'' (from V)
            # i (from W): i_out
            # k (from W): k''? No, W index is k, i.
            # Let's rename W indices to k_w, i_w.
            # W_r[k_w, i_w, c].
            # We need W[k'', i_out, c]. So k_w = k'', i_w = i_out.
            
            # So: U[i', j', c] * V[j'', k'', c] * W[k'', i_out, c]
            # einsum: 'ijc, jkc, kik -> ijkik'
            # i: i'
            # j: j'
            # k: k''
            # i (from W): i_out
            # k (from W): k''? No, the k in W is the same as k in V?
            # Yes, k'' is the index of B's column and C's column.
            # So k_w = k''.
            
            T = np.einsum('ijc,jkc,kic->ijkik', U_r, V_r, W_r)
            
            # Target tensor
            # Target[i', j', j'', k'', i_out, k_out] = 1 if i'=i_out, j'=j'', k''=k_out
            # We can construct this using broadcasting.
            # delta_i = (i' == i_out)
            # delta_j = (j' == j'')
            # delta_k = (k'' == k_out)
            
            # i' is axis 0, i_out is axis 4
            # j' is axis 1, j'' is axis 2
            # k'' is axis 3, k_out is axis 5
            
            delta_i = (np.arange(n) == np.arange(n)).reshape(n, 1, 1, 1, n, 1)
            delta_j = (np.arange(m) == np.arange(m)).reshape(1, m, m, 1, 1, 1)
            delta_k = (np.arange(p) == np.arange(p)).reshape(1, 1, 1, p, 1, p)
            
            Target = delta_i * delta_j * delta_k
            
            error = np.sum((T - Target) ** 2)
            return error

        # Local search
        current_error = compute_error(U, V, W)
        
        # Try a few random perturbations
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and entry
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                row = np.random.randint(n * m)
                col = np.random.randint(target_rank)
                old_val = U[row, col]
                U[row, col] = old_val + np.random.choice([-1, 1])
            elif mat_idx == 1:
                row = np.random.randint(m * p)
                col = np.random.randint(target_rank)
                old_val = V[row, col]
                V[row, col] = old_val + np.random.choice([-1, 1])
            else:
                row = np.random.randint(p * n)
                col = np.random.randint(target_rank)
                old_val = W[row, col]
                W[row, col] = old_val + np.random.choice([-1, 1])
            
            new_error = compute_error(U, V, W)
            
            if new_error < current_error:
                current_error = new_error
            else:
                # Revert
                if mat_idx == 0:
                    U[row, col] = old_val
                elif mat_idx == 1:
                    V[row, col] = old_val
                else:
                    W[row, col] = old_val
        
        if current_error == 0:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break

    # Convert back to lists if needed, but the interface accepts numpy arrays?
    # The seed code returned lists. The interface says "factor matrices of shapes...".
    # Usually numpy arrays are fine, but let's stick to the seed's format if possible.
    # The seed code returned lists of lists.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
