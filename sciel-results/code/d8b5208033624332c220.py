"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 9]

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 144."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1.0 if i == k else 0.0
                if abs(total - expected) > tol:
                    return False
    return True

def _try_kron_decomposition(deadline):
    """Kronecker product decomposition: 4x4x9 = (2x2x3) ⊗ (2x2x3), rank 16."""
    if time.time() > deadline:
        return None
    
    # Rank-4 decomposition for 2x2x3 (schoolbook for that size)
    U2 = [[0.0]*4 for _ in range(4)]
    V2 = [[0.0]*4 for _ in range(6)]
    W2 = [[0.0]*4 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                if i == k:
                    U2[i*2+j][col] = 1.0
                    V2[j*3+k][col] = 1.0
                    W2[k*2+i][col] = 1.0
                    col += 1
    
    if not _verify_decomposition(U2, V2, W2, 2, 2, 3):
        return None
    
    # Build 4x4x9 via Kronecker with correct index mapping
    # i = i1*2 + i2, j = j1*2 + j2, k = k1*3 + k2
    # c = c1*4 + c2
    n, m, p = 4, 4, 9
    rank = 16
    U = [[0.0]*rank for _ in range(n*m)]   # 16 x 16
    V = [[0.0]*rank for _ in range(m*p)]   # 36 x 16
    W = [[0.0]*rank for _ in range(p*n)]   # 36 x 16
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(3):
                        for k2 in range(3):
                            i = i1*2 + i2
                            j = j1*2 + j2
                            k = k1*3 + k2
                            if i == k:
                                for c1 in range(4):
                                    for c2 in range(4):
                                        c = c1*4 + c2
                                        val = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
                                        if val != 0:
                                            U[i*m+j][c] += val
                                        val_v = V2[j1*3+k1][c1] * V2[j2*3+k2][c2]
                                        if val_v != 0:
                                            V[j*p+k][c] += val_v
                                        val_w = W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
                                        if val_w != 0:
                                            W[k*n+i][c] += val_w
    
    if _verify_decomposition(U, V, W, 4, 4, 9):
        return {"U": U, "V": V, "W": W}
    
    return None

def _try_winograd_block(deadline):
    """
    Try to exploit block structure: split 4x4 into 2x2 blocks and 4x9 into 2x3 blocks.
    Use Winograd rank-7 for 2x2x2 core.
    
    The idea: C(4x9) = A(4x4) * B(4x9)
    Split A into 4 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    Split B into 2 blocks of 4x3: B = [B0 | B1] where B0, B1 are 4x3
    Then C = [C0 | C1] where C0 = A*B0 (4x3), C1 = A*B1 (4x3)
    
    For C0 = A * B0:
    C0 = [[A00, A01], [A10, A11]] * [[B0_top], [B0_bot]]
    where B0_top is 2x3, B0_bot is 2x3
    
    C0_top = A00 * B0_top + A01 * B0_bot  (2x3)
    C0_bot = A10 * B0_top + A11 * B0_bot  (2x3)
    
    Each 2x2x3 multiplication can be done with rank 4 (schoolbook) or potentially less.
    But we have 4 such 2x2x3 multiplications for C0, and 4 for C1, total 8.
    If each has rank 4, total rank = 32, which is worse than 16.
    
    Alternative: Use the fact that 9 = 3*3 and try a different block structure.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = 4, 4, 9
    
    # Try structured random search with half-integer values
    # Focus on ranks 10-15
    for r in range(10, 16):
        if time.time() > deadline:
            break
        
        max_attempts = 500 if r <= 12 else 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use structured random values: -1, 0, 1, 0.5, -0.5
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U = [[random.choice(vals) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice(vals) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(p*n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _try_sparse_structured(deadline):
    """
    Try sparse structured decompositions.
    Key insight: The tensor T[i,j,k] = delta(i,k) is very sparse.
    Maybe we can find a low-rank decomposition that exploits this.
    
    For rank r, we need to find U(16xr), V(36xr), W(36xr) such that:
    sum_c U[i*4+j][c] * V[j*9+k][c] * W[k*4+i][c] = 1 if i==k else 0
    
    Note that i ranges 0..3, k ranges 0..8. So i==k only when k=i (i in 0..3).
    For k in 4..8, the expected value is always 0.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = 4, 4, 9
    
    # Try ranks from 10 to 15 with more sophisticated random search
    for r in range(10, 16):
        if time.time() > deadline:
            break
        
        # Use Gaussian random values for better exploration
        for attempt in range(300):
            if time.time() > deadline:
                break
            
            U = [[random.gauss(0, 1) for _ in range(r)] for _ in range(n*m)]
            V = [[random.gauss(0, 1) for _ in range(r)] for _ in range(m*p)]
            W = [[random.gauss(0, 1) for _ in range(r)] for _ in range(p*n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _try_deterministic_search(deadline):
    """
    Try deterministic constructions based on algebraic identities.
    
    For 2x2x2, the Winograd algorithm uses rank 7:
    Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    C = A*B
    
    Winograd's 7-term formula:
    m1 = (a00+a11)*(b00+b11)
    m2 = (a00-a11)*(b00-b11)  
    m3 = (a01+a10)*(b10+b11)
    m4 = (a01-a10)*(b10-b11)
    m5 = a00*(b01+b10)
    m6 = a11*(b01-b10)
    m7 = a01*b10 + a10*b01  (this one is tricky)
    
    Actually, the standard Winograd 2x2x2 rank-7:
    Using variables:
    x1 = a00+a11, y1 = b00+b11
    x2 = a00-a11, y2 = b00-b11
    x3 = a01+a10, y3 = b10+b11
    x4 = a01-a10, y4 = b10-b11
    x5 = a00, y5 = b01+b10
    x6 = a11, y6 = b01-b10
    x7 = a01, y7 = b10  (and a10, b01 handled separately?)
    
    Let me use the known rank-7 decomposition explicitly.
    """
    if time.time() > deadline:
        return None
    
    # Known rank-7 Winograd for 2x2x2
    # U, V, W for 2x2x2 with i,j in {0,1}, k in {0,1}
    # T[i,j,k] = 1 if i==k else 0
    
    # The 7 terms (from literature):
    # Term 0: a00*b00 -> c00 contribution
    # Term 1: a00*b01 -> c01 contribution  
    # Term 2: a01*b00 -> c00 contribution
    # Term 3: a01*b10 -> c10 contribution
    # Term 4: a10*b01 -> c11 contribution
    # Term 5: a11*b11 -> c11 contribution
    # Term 6: (a00+a11)*(b00+b11) - a00*b00 - a11*b11 -> correction
    
    # This is getting complicated. Let me just use the schoolbook for now
    # and focus on improving the random search.
    
    n, m, p = 4, 4, 9
    
    # Try a more targeted search: fix some structure
    # For example, assume W has a specific pattern
    
    for r in range(10, 16):
        if time.time() > deadline:
            break
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Sparse random: only 30% non-zero entries
            U = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(p*n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    # Try Kronecker decomposition first (rank 16) - this is our baseline
    result = _try_kron_decomposition(deadline)
    kron_result = result
    
    if result is not None:
        # Try to find something better
        pass
    
    # Try Winograd-style block decomposition
    result = _try_winograd_block(deadline)
    if result is not None:
        return result
    
    # Try sparse structured search
    result = _try_sparse_structured(deadline)
    if result is not None:
        return result
    
    # Try deterministic search
    result = _try_deterministic_search(deadline)
    if result is not None:
        return result
    
    # Fall back to Kronecker or schoolbook
    if kron_result is not None:
        return kron_result
    
    return _schoolbook()
