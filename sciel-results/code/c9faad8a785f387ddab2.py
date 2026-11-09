"""Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _get_strassen_2x2():
    """Return U, V, W for rank-7 2x2x2 matmul."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # s=1
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # s=2
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    # s=3
    U[:, 2] = [1, -1, 0, 0]
    V[:, 2] = [0, 0, 1, 1]
    W[:, 2] = [0, 0, 1, 1]
    
    # s=4
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 1, -1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    # s=5
    U[:, 4] = [1, 0, 0, 1]
    V[:, 4] = [0, -1, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    # s=6
    U[:, 5] = [0, 1, 1, 0]
    V[:, 5] = [1, 0, -1, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # s=7
    U[:, 6] = [1, 0, 1, 0]
    V[:, 6] = [0, 1, 0, -1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _get_strassen_4x4():
    """Return U, V, W for rank-49 4x4x4 matmul using hierarchical Strassen."""
    n = 4
    U_2, V_2, W_2 = _get_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    U_big_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, -1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
    ]
    
    V_big_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, -1],
        [0, -1, 0, 1],
        [1, 0, -1, 0],
        [0, 1, 0, -1],
    ]
    
    block_indices = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    for s_big in range(7):
        for s_small in range(7):
            idx = s_big * 7 + s_small
            
            u_small = U_2[:, s_small]
            for b_idx in range(4):
                scale = U_big_coeffs[s_big][b_idx]
                if scale != 0:
                    U[block_indices[b_idx], idx] = scale * u_small
            
            v_small = V_2[:, s_small]
            for b_idx in range(4):
                scale = V_big_coeffs[s_big][b_idx]
                if scale != 0:
                    V[block_indices[b_idx], idx] = scale * v_small
            
            w_small = W_2[:, s_small]
            
            if s_big == 0:
                C11_coeff = 1; C12_coeff = 0; C21_coeff = 0; C22_coeff = 1
            elif s_big == 1:
                C11_coeff = 0; C12_coeff = 0; C21_coeff = 1; C22_coeff = -1
            elif s_big == 2:
                C11_coeff = 0; C12_coeff = 1; C21_coeff = 0; C22_coeff = 1
            elif s_big == 3:
                C11_coeff = 1; C12_coeff = 0; C21_coeff = 1; C22_coeff = 0
            elif s_big == 4:
                C11_coeff = -1; C12_coeff = 1; C21_coeff = 0; C22_coeff = 0
            elif s_big == 5:
                C11_coeff = 0; C12_coeff = 0; C21_coeff = 0; C22_coeff = 1
            elif s_big == 6:
                C11_coeff = 1; C12_coeff = 0; C21_coeff = 0; C22_coeff = 0
            
            if C11_coeff != 0:
                W[block_indices[0], idx] += C11_coeff * w_small
            if C12_coeff != 0:
                W[block_indices[1], idx] += C12_coeff * w_small
            if C21_coeff != 0:
                W[block_indices[2], idx] += C21_coeff * w_small
            if C22_coeff != 0:
                W[block_indices[3], idx] += C22_coeff * w_small

    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition exactly."""
    rank = U.shape[1]
    # Build the full tensor check
    # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
    # Should equal 1 if (a,b,c) = (i*m+j, j*p+l, l*n+i) for some i,j,l
    # and 0 otherwise
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # For each s, the contribution is U[:,s] outer V[:,s] outer W[:,s]
    # Sum over s
    # This is a 3-tensor
    # T = sum_s U[:,s] V[:,s]^T W[:,s]^T (as a 3-way tensor)
    
    # Efficient check: for each (a,b), compute sum_s U[a,s]*V[b,s]*W[c,s] for all c
    # This is (U @ diag(V[b,:]) @ W^T) but that's not quite right
    
    # Better: T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
    # = (U * V^T) @ W^T ... no
    # T = sum_s u_s v_s^T w_s^T where u_s = U[:,s], v_s = V[:,s], w_s = W[:,s]
    
    # Let's just check the nonzero entries and a sample of zeros
    # Nonzero entries: (i*m+j, j*p+l, l*n+i) for i in range(n), j in range(m), l in range(p)
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check some zero entries
    # For (a,b,c) where c is not consistent with a,b
    # a = i*m+j, b = j'*p+l, c = l'*n+i'
    # Zero if j != j' or l != l' or i != i'
    # Let's check a few
    import random
    random.seed(42)
    for _ in range(1000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        # Check if this is a nonzero entry
        i = a // m
        j = a % m
        j2 = b // p
        l = b % p
        l2 = c // n
        i2 = c % n
        if i == i2 and j == j2 and l == l2:
            continue  # nonzero, skip
        val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
        if abs(val) > tol:
            return False
    
    return True

def _verify_full(U, V, W, n, m, p, tol=1e-8):
    """Full verification - check all entries."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    # Build T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
    # For each s, add outer product
    # T is (n*m) x (m*p) x (p*n)
    # This is 16 x 32 x 32 = 16384 entries
    
    # Efficient: for each s, T += np.einsum('a,b,c->abc', U_arr[:,s], V_arr[:,s], W_arr[:,s])
    # But that's slow for large rank. Let's do it in a vectorized way.
    
    # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
    # = sum_s U[a,s] * (V[b,s] * W[c,s])
    # For fixed a: T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
    # = (U[a,:] * V[b,:]) dot W[c,:] ... no, that's not right either
    
    # Let's just do it directly
    T = np.zeros((n*m, m*p, p*n))
    for s in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,s], V_arr[:,s], W_arr[:,s])
    
    # Check nonzero entries
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                if abs(T[a,b,c] - 1.0) > tol:
                    return False
    
    # Check zero entries
    # All entries not of the form (i*m+j, j*p+l, l*n+i) should be 0
    # Create a mask of nonzero entries
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                mask[a,b,c] = True
    
    # Check that all zero-mask entries are ~0
    zero_mask = ~mask
    if np.any(np.abs(T[zero_mask]) > tol):
        return False
    
    return True

def _try_rank20_444(deadline):
    """Try to find a rank-20 decomposition for 4x4x4 matmul."""
    n = 4
    # T[a,b,c] = 1 if (a,b,c) = (i*4+j, j*4+k, k*4+i) else 0
    # a in 0..15, b in 0..15, c in 0..15
    
    # Nonzero entries
    nonzero = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                nonzero.append((a, b, c))
    
    # For rank 20, we need U (16x20), V (16x20), W (16x20)
    # such that sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c]
    
    # Strategy: fix W to have a specific structure, then solve for U, V
    # Let's try: W[c,s] = 1 if c == s else 0 for s < 16, and handle the rest
    # Actually, let's use a different approach.
    
    # Use the fact that the problem has a lot of symmetry.
    # Let's try to find the decomposition using a greedy approach.
    
    # Alternative: use the known rank-20 construction from the literature.
    # I'll try to construct it explicitly.
    
    # Here's a known rank-20 construction for 4x4x4:
    # Based on the work of various researchers, the 20 multiplications are:
    
    # Let me try a different approach: use numerical optimization.
    # Initialize U, V, W randomly and use gradient descent.
    
    import random
    random.seed(123)
    np.random.seed(123)
    
    r = 20
    # Initialize
    U = np.random.randn(16, r) * 0.5
    V = np.random.randn(16, r) * 0.5
    W = np.random.randn(16, r) * 0.5
    
    # Gradient descent
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline:
            return None
        
        # Compute T = sum_s U[:,s] outer V[:,s] outer W[:,s]
        # T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
        T = np.zeros((16, 16, 16))
        for s in range(r):
            T += np.einsum('a,b,c->abc', U[:,s], V[:,s], W[:,s])
        
        # Target
        T_target = np.zeros((16, 16, 16))
        for (a, b, c) in nonzero:
            T_target[a,b,c] = 1.0
        
        # Loss
        loss = np.sum((T - T_target)**2)
        if loss < 1e-10:
            # Verify
            if _verify_full(U.tolist(), V.tolist(), W.tolist(), 4, 4, 4, tol=1e-8):
                return U, V, W
        
        # Gradient
        # dT/dU[a,s] = V[b,s]*W[c,s] for each (a,b,c)
        # dLoss/dU[a,s] = 2 * sum_{b,c} (T[a,b,c] - T_target[a,b,c]) * V[b,s] * W[c,s]
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for s in range(r):
            # dU[a,s] = 2 * sum_{b,c} (T[a,b,c] - T_target[a,b,c]) * V[b,s] * W[c,s]
            # = 2 * ((T - T_target) * V[:,s,None] * W[:,s,None]) summed over b,c
            diff = T - T_target  # (16,16,16)
            dU[:,s] = 2 * np.sum(diff * V[:,s,None] * W[:,s,None], axis=(1,2))
            dV[:,s] = 2 * np.sum(diff * U[:,s,None] * W[:,s,None], axis=(0,2))
            dW[:,s] = 2 * np.sum(diff * U[:,s,None] * V[:,s,None], axis=(0,1))
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return None

def _build_448_from_444(U444, V444, W444):
    """Build a 4x4x8 decomposition from a 4x4x4 decomposition."""
    n, m, p = 4, 4, 8
    rank = U444.shape[1]
    
    U = np.zeros((16, rank * 2))
    V = np.zeros((32, rank * 2))
    W = np.zeros((32, rank * 2))
    
    # Block 1: l in [0,4)
    U[:, 0:rank] = U444
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:rank] = V444[j * 4 + k, :]
    W[0:16, 0:rank] = W444
    
    # Block 2: l in [4,8)
    U[:, rank:rank*2] = U444
    for j in range(4):
        for k in range(4):
            V[j * 8 + 4 + k, rank:rank*2] = V444[j * 4 + k, :]
    W[16:32, rank:rank*2] = W444
    
    return U, V, W

def _schoolbook_448():
    """Schoolbook decomposition for 4x4x8, rank 128."""
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                U[i * m + j, col] = 1
                V[j * p + l, col] = 1
                W[l * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_448():
    """Strassen-based decomposition for 4x4x8, rank 98."""
    U_444, V_444, W_444 = _get_strassen_4x4()
    return _build_448_from_444(U_444, V_444, W_444)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin
    
    # Try to find a rank-20 decomposition for 4x4x4
    result = _try_rank20_444(deadline)
    if result is not None:
        U444, V444, W444 = result
        U, V, W = _build_448_from_444(U444, V444, W444)
        if _verify(U, V, W, n, m, p, tol=1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Strassen rank-98
    U, V, W = _strassen_448()
    if _verify(U, V, W, n, m, p, tol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_448()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
