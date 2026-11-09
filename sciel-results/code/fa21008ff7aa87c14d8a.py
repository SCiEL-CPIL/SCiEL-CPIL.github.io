import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    size_a = n * m
    size_b = m * p
    size_c = p * n
    
    # Build the multiplication tensor T[a, b, c]
    # a = i*m + j, b = j'*p + k, c = k'*n + i
    # T[a, b, c] = 1 if j(a) == j'(b) and i(a) == i'(c) and k(b) == k'(c)
    T = np.zeros((size_a, size_b, size_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # We'll use a greedy peeling approach over the rationals
    # At each step, find the best rank-1 approximation of the remaining tensor
    # using a heuristic based on the largest entries
    
    T_curr = T.copy().astype(np.float64)
    
    U_list = []
    V_list = []
    W_list = []
    
    start_time = time.time()
    max_iterations = 200
    
    for r in range(max_iterations):
        if time.time() - start_time > 110:
            break
        
        # Check if T_curr is zero
        if np.max(np.abs(T_curr)) < 1e-10:
            break
        
        # Find the entry with maximum absolute value
        max_idx = np.unravel_index(np.argmax(np.abs(T_curr)), T_curr.shape)
        a_idx, b_idx, c_idx = max_idx
        max_val = T_curr[a_idx, b_idx, c_idx]
        
        # Build a rank-1 term that matches the structure of the tensor
        # The tensor has the property that T[a, b, c] = 1 when:
        # a = i*m + j, b = j*p + k, c = k'*n + i' where j=j', i=i', k=k'
        # So for a given (a_idx, b_idx, c_idx), we can extract:
        # i = a_idx // m, j = a_idx % m
        # j' = b_idx // p, k = b_idx % p
        # i' = c_idx % n, k' = c_idx // n
        
        i = a_idx // m
        j = a_idx % m
        j_prime = b_idx // p
        k = b_idx % p
        i_prime = c_idx % n
        k_prime = c_idx // n
        
        # For a valid tensor entry, we should have j == j_prime, i == i_prime, k == k_prime
        # If not, this is a "wrong" entry that needs to be cancelled
        
        # Strategy: create a rank-1 term that covers all entries with the same
        # (i, j, k) pattern. This is essentially the schoolbook approach but
        # we try to be smarter by looking at the structure.
        
        # Let's use a different approach: use the SVD of the matricizations
        # to find the best rank-1 approximation
        
        # Matricize along the first mode: M1 is size_a x (size_b * size_c)
        M1 = T_curr.reshape(size_a, -1)
        
        # Get the top left singular vector
        try:
            U_svd, s_vals, Vt = np.linalg.svd(M1, full_matrices=False)
            u_vec = U_svd[:, 0]
        except:
            # Fallback: use the spike at the max entry
            u_vec = np.zeros(size_a)
            u_vec[a_idx] = 1.0
        
        # Now find v_vec and w_vec from the residual
        # The rank-1 term is u_vec \otimes v_vec \otimes w_vec
        # We need to find v_vec and w_vec such that:
        # T_curr[a_idx, :, :] ≈ u_vec[a_idx] * v_vec \otimes w_vec
        
        # Matricize the slice at a_idx: T_curr[a_idx, :, :] is size_b x size_c
        slice_ac = T_curr[a_idx, :, :]
        
        # Scale by u_vec[a_idx]
        if abs(u_vec[a_idx]) > 1e-10:
            slice_scaled = slice_ac / u_vec[a_idx]
        else:
            slice_scaled = slice_ac
        
        # SVD of slice_scaled to find v_vec and w_vec
        try:
            U_slice, s_slice, Vt_slice = np.linalg.svd(slice_scaled, full_matrices=False)
            v_vec = U_slice[:, 0]
            w_vec = Vt_slice[0, :]
        except:
            # Fallback: spike at max entry in the slice
            v_vec = np.zeros(size_b)
            w_vec = np.zeros(size_c)
            v_idx, w_idx = np.unravel_index(np.argmax(np.abs(slice_scaled)), slice_scaled.shape)
            v_vec[v_idx] = 1.0
            w_vec[w_idx] = 1.0
        
        # Scale the vectors to match the magnitude
        # The term should be u_vec \otimes v_vec \otimes w_vec
        # We want T_curr[a_idx, b_idx, c_idx] to be approximately matched
        # u_vec[a_idx] * v_vec[b_idx] * w_vec[c_idx] ≈ max_val
        
        if abs(u_vec[a_idx] * v_vec[b_idx] * w_vec[c_idx]) > 1e-10:
            scale = max_val / (u_vec[a_idx] * v_vec[b_idx] * w_vec[c_idx])
        else:
            scale = 1.0
        
        u_term = u_vec * scale
        v_term = v_vec
        w_term = w_vec
        
        # Subtract the rank-1 term from T_curr
        for a in range(size_a):
            if abs(u_term[a]) < 1e-10:
                continue
            for b in range(size_b):
                if abs(v_term[b]) < 1e-10:
                    continue
                for c in range(size_c):
                    if abs(w_term[c]) < 1e-10:
                        continue
                    T_curr[a, b, c] -= u_term[a] * v_term[b] * w_term[c]
        
        U_list.append(u_term)
        V_list.append(v_term)
        W_list.append(w_term)
    
    # Check if we achieved zero residual
    if len(U_list) > 0 and np.max(np.abs(T_curr)) < 1e-10:
        rank = len(U_list)
        # Convert to lists of floats (or fractions if possible)
        U = [[float(x) for x in row] for row in U_list]
        V = [[float(x) for x in row] for row in V_list]
        W = [[float(x) for x in row] for row in W_list]
        # Transpose to get shape (size_a, rank), etc.
        U = np.array(U).T.tolist()
        V = np.array(V).T.tolist()
        W = np.array(W).T.tolist()
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(size_a)]
    V = [[0] * rank for _ in range(size_b)]
    W = [[0] * rank for _ in range(size_c)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
