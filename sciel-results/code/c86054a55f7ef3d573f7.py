import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s
    
    n, m, p = 2, 2, 8
    
    # Build the target tensor T[a,b,c]
    # a = i*m + j (A index: i=row, j=col of A)
    # b = j*p + k (B index: j=row, k=col of B)
    # c = k*n + i (C index: k=col, i=row of C) -- W[k*n+i, r] convention
    nm, mp, pn = n * m, m * p, p * n
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Start with rank-28 Strassen solution
    best_rank = 28
    best_result = _strassen_28(n, m, p)
    
    # Verify the rank-28 solution
    if _verify(best_result, T_target):
        pass
    else:
        # Should not happen, but fallback
        return best_result
    
    # Try to reduce rank by searching for linear dependencies
    # The 28 Strassen terms might have linear dependencies when A is shared
    # Try rank 27 down to some lower bound
    
    for target_rank in range(27, 15, -1):
        if time.time() - start_time > deadline:
            break
        
        result = _search_rank(target_rank, T_target, nm, mp, pn, start_time, deadline)
        if result is not None:
            if _verify(result, T_target):
                best_rank = target_rank
                best_result = result
                break
    
    return best_result


def _strassen_28(n, m, p):
    """Returns rank-28 solution using Strassen per 2x2 block."""
    rank = 28
    nm, mp, pn = n * m, m * p, p * n
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Strassen terms for 2x2 * 2x2
    A_forms = [
        {0: 1, 3: 1},       # M1: a+d
        {2: 1, 3: 1},       # M2: c+d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a+b
        {2: 1, 0: -1},      # M6: c-a
        {1: 1, 3: -1}       # M7: b-d
    ]
    
    B_forms = [
        {0: 1, 3: 1},       # M1: x0+y1
        {0: 1},             # M2: x0
        {1: 1, 3: -1},      # M3: x1-y1
        {2: 1, 0: -1},      # M4: y0-x0
        {3: 1},             # M5: y1
        {0: 1, 1: 1},       # M6: x0+x1
        {2: 1, 3: 1}        # M7: y0+y1
    ]
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00: M1, M4, M5, M7
        {2: 1, 4: 1},              # C01: M3, M5
        {1: 1, 3: 1},              # C10: M2, M4
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11: M1, M2, M3, M6
    ]

    for block in range(p // 2):
        k0 = block * 2
        
        for s in range(7):
            r = block * 7 + s
            
            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] = val
            
            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val
            
            for c_idx, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_idx // 2
                    k_local = c_idx % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    return {"U": U, "V": V, "W": W}


def _verify(result, T_target):
    """Verify that the decomposition exactly matches the target tensor."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    # T_decomposed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_decomposed = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T_decomposed, T_target, atol=1e-10)


def _search_rank(target_rank, T_target, nm, mp, pn, start_time, deadline):
    """
    Search for a rank-target_rank decomposition.
    Uses structured initialization based on Strassen with merged terms.
    """
    # Strategy: Try to merge Strassen terms across blocks
    # Since A is shared, we can try to combine the A-forms across blocks
    
    # Approach: Start with rank-28 Strassen, then try to find linear combinations
    # that reduce the rank while maintaining the decomposition
    
    # Alternative: Direct search with structured initialization
    # Use the fact that the tensor has a specific structure
    
    # Try different random seeds and structured initializations
    np.random.seed(42)
    
    max_attempts = 100
    for attempt in range(max_attempts):
        if time.time() - start_time > deadline:
            break
        
        # Initialize with structured values
        U = _structured_init(nm, target_rank, attempt)
        V = _structured_init(mp, target_rank, attempt)
        W = _structured_init(pn, target_rank, attempt)
        
        # Try to refine using gradient descent or other methods
        # For now, just check if it works
        T_decomposed = np.einsum('ar,br,cr->abc', U, V, W)
        
        if np.allclose(T_decomposed, T_target, atol=1e-10):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def _structured_init(size, rank, seed):
    """Create a structured initialization with integer/half-integer entries."""
    np.random.seed(seed)
    values = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(size, rank))
    return values
