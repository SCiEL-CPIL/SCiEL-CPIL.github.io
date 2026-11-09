"""Matrix multiplication 3x3x5: attempt rank-26 via Laderman extension, fallback to schoolbook 45."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

DEADLINE = 150.0  # wall-clock limit
SAFETY_MARGIN = 10.0  # seconds to leave before deadline


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-9):
    """Check that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Reconstruct the 3D tensor
    # T[i,j,k] should equal delta (i.e., T[i,j,k] = 1 if this is the multiplication tensor entry)
    # Actually the multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in the "middle" 
    # The tensor for matmul: T_{i,j,k} where output[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k)
    # Reconstruction: T_recon[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Note W index: W[k*n+i, r]
    T_recon = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    # The multiplication tensor is all ones
    T_target = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(T_recon, T_target, atol=tol)


def _try_laderman_extension():
    """
    Attempt to construct a rank-26 decomposition for 3x3x5 by extending Laderman's method.
    
    Laderman's 3x3x3 uses 7 multiplications. The idea for 3x3x5:
    We compute C = A * B where A is 3x3, B is 3x5, C is 3x5.
    
    The tensor has 3*3*5 = 45 entries. We want to decompose it into rank r.
    
    Strategy: Use the 7 Laderman multiplications for the "left 3 columns" of B,
    and extend with additional multiplications for columns 4 and 5.
    
    Actually, a cleaner approach: think of it as 3x3x3 + 3x3x2.
    The 3x3x3 part uses rank 7 (Laderman). The 3x3x2 part needs rank at most 3*3*2=18,
    but we can share. Total naive: 7+18=25, but we need to be careful.
    
    Let me try a direct algebraic construction.
    
    For 3x3x3, Laderman's 7 products are:
    M1 = (A00 - A01)(B10 + B11)
    M2 = (A00 + A01)(B10 - B11)
    M3 = (A00 - A10)(B01 + B11)
    M4 = (A00 + A10)(B01 - B11)
    M5 = (A11 - A01)(B00 + B10)
    M6 = (A11 + A01)(B00 - B10)
    M7 = (A11 - A10)(B01 + B11)
    
    For the 3x3x5 case, B has 5 columns. Let me denote B columns as B_0..B_4.
    We need C_0..C_4 where C_k = A * B_k.
    
    Approach: Use Laderman for computing A*B_0 and A*B_1 (giving C_0, C_1),
    then use additional multiplications for B_2, B_3, B_4.
    
    For each additional column B_k (k=2,3,4), we need 3x3 matrix multiplication
    which naively takes 9 products, but we can use Laderman's 7.
    
    However, to keep total rank <= 26, we need:
    - 7 for first 3x3x3 block (columns 0,1) -- wait, that's only 2 columns
    
    Let me reconsider. Laderman handles 3x3x3 = computing a 3x3 output from 3x3 * 3x3.
    For 3x3x5, we compute 3x5 output.
    
    Better framing: The tensor T[i,j,k] for i in 0..2, j in 0..2, k in 0..4.
    
    Let me try a structured approach with rank 26:
    - Use 7 Laderman-like products that handle "most" of the computation
    - Add 19 more products to complete the remaining terms
    
    Actually, let me try a more systematic approach.
    
    For a 3x3x5 multiplication, one approach is to note that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can write this using rank-r decomposition where each rank-1 term
    is a product of a linear form in A, a linear form in B, and a linear form in C.
    
    Let me try to construct this explicitly. I'll use a greedy/structured approach:
    
    For columns k=0,1: use Laderman's 7 products (adapted)
    For columns k=2,3,4: use 3 additional products each (total 9 more)
    Total: 7 + 9 = 16? That seems too optimistic.
    
    Let me be more careful. For a single column B_k, computing A*B_k (a 3x1 vector)
    requires 3*3 = 9 products naively, or 7 with Laderman-like tricks.
    
    But we have 5 columns. If we use 7 per column: 5*7 = 35. Not great.
    
    Alternative: group columns. 
    - Columns 0,1: treat as 3x3x2 sub-problem
    - Columns 2,3,4: treat as 3x3x3 sub-problem
    
    For 3x3x3: rank 7 (Laderman)
    For 3x3x2: rank at most 6 (using 2 Laderman-like sets? or 3x3x2 can be done in rank 6?)
    
    Actually 3x3x2: the tensor has 18 entries. Rank is at most 18 (schoolbook).
    Can we do better? The border rank / rank of 3x3x2...
    
    Let me just try to construct a valid rank-26 decomposition directly.
    
    I'll use the following strategy:
    - 7 products for the "core" 3x3x3 (handling columns 0,1 and part of column 2)
    - 19 additional products to handle the rest
    
    This is getting complex. Let me try a different approach:
    directly verify if a specific known construction works.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Let's try a structured construction with rank 26
    # We'll define U, V, W as 26-column matrices and try to fill them.
    
    # Approach: Use 7 Laderman products for columns 0-1 (3x3x2 part)
    # and 19 products for columns 2-4 (3x3x3 part)
    # Total: 26
    
    # Laderman for 3x3x3 (computing C = A*B where A,B are 3x3, C is 3x3):
    # The 7 products and their contributions to C[i,k]:
    # C[0,0] = M1 - M2 + M5 - M6  ... etc.
    # Let me use the standard Laderman decomposition.
    
    # Standard Laderman (from literature):
    # M1 = (A00 - A01) * (B10 + B11)
    # M2 = (A00 + A01) * (B10 - B11)  
    # M3 = (A00 - A10) * (B01 + B11)
    # M4 = (A00 + A10) * (B01 - B11)
    # M5 = (A11 - A01) * (B00 + B10)
    # M6 = (A11 + A01) * (B00 - B10)
    # M7 = (A11 - A10) * (B01 + B11)
    
    # C[0,0] = M1 - M2 + M5 - M6  ... I need to look this up properly.
    
    # Actually, let me use a well-known formulation. The Laderman algorithm:
    # p1 = (a00 - a01)(b10 + b11)
    # p2 = (a00 + a01)(b10 - b11)
    # p3 = (a00 - a10)(b01 + b11)
    # p4 = (a00 + a10)(b01 - b11)
    # p5 = (a11 - a01)(b00 + b10)
    # p6 = (a11 + a01)(b00 - b10)
    # p7 = (a11 - a10)(b01 + b11)
    
    # c00 = p1 - p2 + p5 - p6
    # c01 = p3 - p4 + p7
    # c02 = p3 + p4 - p7  -- wait, this doesn't seem right for 3x3
    
    # Let me just use a verified Laderman decomposition.
    # From various sources, the 7 multiplications for 3x3x3:
    
    # I'll use a different, well-documented version:
    # The tensor T[i,j,k] for 3x3x3 matmul can be written as sum of 7 rank-1 terms.
    
    # Let me just try to verify a specific candidate and if it doesn't work,
    # fall back to schoolbook.
    
    # For now, let me try a simpler approach: 
    # Use schoolbook for the 3x3x2 part (rank 18) and Laderman for 3x3x3 part (rank 7)
    # Total: 18 + 7 = 25. But we need to handle the overlap correctly.
    
    # Actually, 3x3x2 + 3x3x3 = 3x3x5 if we partition columns.
    # Columns 0,1 form the 3x3x2 part (18 entries, rank <= 18)
    # Columns 2,3,4 form the 3x3x3 part (27 entries, rank <= 7 via Laderman)
    # Total rank <= 18 + 7 = 25
    
    # But wait, the 3x3x2 part with schoolbook is rank 18, and Laderman for 3x3x3 is rank 7.
    # Total: 25. That's better than 45!
    
    # Let me construct this:
    # Part 1: Schoolbook for columns 0,1 (3x3x2)
    # Part 2: Laderman for columns 2,3,4 (3x3x3, relabeled)
    
    rank = 25
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Part 1: Schoolbook for B columns 0,1 (k=0,1)
    # For each (i,j,k) with k in {0,1}:
    #   U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(2):  # columns 0,1
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    # col is now 18
    
    # Part 2: Laderman for B columns 2,3,4 (treated as a 3x3x3 problem)
    # Relabel: B' = B[:, 2:5], so B'[j, k'] = B[j, k'+2] where k' in {0,1,2}
    # C'[i, k'] = C[i, k'+2]
    # A is unchanged.
    
    # Laderman's 7 products for 3x3x3:
    # I'll use the standard Laderman decomposition.
    # Reference: Laderman, "A matrix multiplication algorithm in seven multiplications"
    
    # The 7 rank-1 terms for 3x3x3 matmul (A 3x3, B 3x3, C 3x3):
    # Let me define them carefully.
    
    # From the literature, one common formulation:
    # m1 = (a00 - a01)(b10 + b11)
    # m2 = (a00 + a01)(b10 - b11)
    # m3 = (a00 - a10)(b01 + b11)
    # m4 = (a00 + a10)(b01 - b11)
    # m5 = (a11 - a01)(b00 + b10)
    # m6 = (a11 + a01)(b00 - b10)
    # m7 = (a11 - a10)(b01 + b11)
    
    # c00 = m1 - m2 + m5 - m6
    # c01 = m3 - m4 + m7
    # c02 = m3 + m4 - m7  -- Hmm, I'm not confident in the exact coefficients.
    
    # Let me use a verified set. Actually, let me just try to verify
    # a candidate Laderman decomposition programmatically.
    
    # I'll define the 7 terms and check if they reconstruct the 3x3x3 tensor.
    
    # Laderman terms (U, V, W vectors for each term):
    # For term r, we have:
    #   u_r[i,j] = coefficient of A[i,j] in the first factor
    #   v_r[j,k] = coefficient of B[j,k] in the second factor
    #   w_r[k,i] = coefficient of C[i,k] in the output (note: w indexed as [k*n+i])
    
    # Let me define the 7 Laderman terms:
    # Using a standard reference implementation:
    
    laderman_terms = [
        # (u: 3x3, v: 3x3, w: 3x3) where w[k,i] is the coefficient
        # Term 1: (a00 - a01) * (b10 + b11)
        (np.array([[1, -1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
         np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float64),
         None),  # w to be determined
        # Term 2: (a00 + a01) * (b10 - b11)
        (np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
         np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float64),
         None),
        # Term 3: (a00 - a10) * (b01 + b11)
        (np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 0]], dtype=np.float64),
         np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64),
         None),
        # Term 4: (a00 + a10) * (b01 - b11)
        (np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64),
         np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float64),
         None),
        # Term 5: (a11 - a01) * (b00 + b10)
        (np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64),
         np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64),
         None),
        # Term 6: (a11 + a01) * (b00 - b10)
        (np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64),
         np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 0]], dtype=np.float64),
         None),
        # Term 7: (a11 - a10) * (b01 + b11)
        (np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float64),
         np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64),
         None),
    ]
    
    # Now I need to find w_r such that sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all (i,j,k)
    # This means: for each (i,j,k), sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    
    # Let me set up a linear system for w.
    # Unknowns: w_r[k,i] for r=0..6, k=0..2, i=0..2 => 7*9 = 63 unknowns
    # Equations: for each (i,j,k) in 3x3x3 => 27 equations
    # This is underdetermined, so I need to find ANY solution.
    
    # Let me set up the system:
    # For each (i,j,k): sum_{r=0}^{6} u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    
    # Let me flatten w_r[k,i] into a vector of length 63.
    # Index: idx = r*9 + k*3 + i
    
    A_sys = np.zeros((27, 63), dtype=np.float64)
    b_sys = np.ones(27, dtype=np.float64)
    
    eq = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for r in range(7):
                    u_val = laderman_terms[r][0][i, j]
                    v_val = laderman_terms[r][1][j, k]
                    if u_val != 0 and v_val != 0:
                        idx = r * 9 + k * 3 + i
                        A_sys[eq, idx] = u_val * v_val
                eq += 1
    
    # Solve the underdetermined system: find minimum-norm solution
    try:
        w_sol = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
        # Verify
        recon = A_sys @ w_sol
        if np.allclose(recon, b_sys, atol=1e-9):
            # Great! We have a valid Laderman decomposition.
            # Now fill in the W matrix for the 3x3x3 part.
            for r in range(7):
                col_idx = 18 + r
                for k in range(3):
                    for i in range(3):
                        w_val = w_sol[r * 9 + k * 3 + i]
                        # In our 3x3x5 problem, column k' = k corresponds to actual column k+2
                        # W index: [k_actual * n + i, col]
                        W[(k + 2) * n + i, col_idx] = w_val
                # Fill U and V for this term
                U[:, col_idx] = laderman_terms[r][0].flatten()
                # V: for the 3x3x3 part, B' column k' maps to actual column k+2
                v_mat = laderman_terms[r][1]
                for j in range(3):
                    for kp in range(3):
                        V[j * p + (kp + 2), col_idx] = v_mat[j, kp]
            
            # Verify the full 3x3x5 decomposition
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    return None


def _try_random_search(max_rank=30, max_time=120.0):
    """Try to find a decomposition with rank < 45 using random search."""
    start = time.time()
    n, m, p = DIMENSIONS
    
    for rank in range(26, min(max_rank + 1, 45)):
        if time.time() - start > max_time:
            break
        for attempt in range(50):
            if time.time() - start > max_time:
                break
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            # Normalize
            for r in range(rank):
                norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
                if norm > 1e-12:
                    scale = 1.0 / (norm ** (1.0 / 3.0))
                    U[:, r] *= scale
                    V[:, r] *= scale
                    W[:, r] *= scale
            
            if _verify(U, V, W, tol=1e-6):
                return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    start_time = time.time()
    
    # First, try the Laderman extension (rank 25)
    result = _try_laderman_extension()
    if result is not None:
        return result
    
    # Check time
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        return _schoolbook()
    
    # Try random search for rank < 45
    result = _try_random_search(max_rank=35, max_time=DEADLINE - SAFETY_MARGIN - (time.time() - start_time))
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook()
