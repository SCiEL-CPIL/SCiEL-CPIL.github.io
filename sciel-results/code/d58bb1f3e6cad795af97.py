import time
import numpy as np

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 145
_START_TIME = time.time()

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify_fast(U, V, W, n, m, p):
    rank = U.shape[1]
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        row_A = i // m
        col_A = i % m
        for j in range(m*p):
            row_B = j // p
            col_B = j % p
            if col_A == row_B:
                k = col_B * n + row_A
                T_expected[i, j, k] = 1.0
    T_actual = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_actual, T_expected, atol=1e-9)

def _construct_rank55():
    n, m, p = DIMENSIONS
    rank = 55
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02 = 2
    a12 = 5
    for b in range(5):
        k0 = 2*b
        k1 = 2*b+1
        b00, b01 = k0, k1
        b10, b11 = 10+k0, 10+k1
        b20, b21 = 20+k0, 20+k1
        c00, c01 = k0*2, k1*2
        c10, c11 = k0*2+1, k1*2+1
        # M1: (A00+A11)(B00+B11)
        U[a00,col]=1; U[a11,col]=1; V[b00,col]=1; V[b11,col]=1; W[c00,col]=1; W[c11,col]=1
        col+=1
        # M2: (A10+A11)B00
        U[a10,col]=1; U[a11,col]=1; V[b00,col]=1; W[c10,col]=1; W[c11,col]=-1
        col+=1
        # M3: A00(B01-B11)
        U[a00,col]=1; V[b01,col]=1; V[b11,col]=-1; W[c01,col]=1; W[c11,col]=1
        col+=1
        # M4: A11(B10-B00)
        U[a11,col]=1; V[b10,col]=1; V[b00,col]=-1; W[c00,col]=1; W[c10,col]=1
        col+=1
        # M5: (A00+A01)B11
        U[a00,col]=1; U[a01,col]=1; V[b11,col]=1; W[c00,col]=-1; W[c01,col]=1
        col+=1
        # M6: (A10-A00)(B00+B01)
        U[a10,col]=1; U[a00,col]=-1; V[b00,col]=1; V[b01,col]=1; W[c11,col]=1
        col+=1
        # M7: (A01-A11)(B10+B11)
        U[a01,col]=1; U[a11,col]=-1; V[b10,col]=1; V[b11,col]=1; W[c00,col]=1
        col+=1
        # P1-P4: schoolbook for A2*B2
        U[a02,col]=1; V[b20,col]=1; W[c00,col]=1; col+=1
        U[a02,col]=1; V[b21,col]=1; W[c01,col]=1; col+=1
        U[a12,col]=1; V[b20,col]=1; W[c10,col]=1; col+=1
        U[a12,col]=1; V[b21,col]=1; W[c11,col]=1; col+=1
    return {'U': U, 'V': V, 'W': W}

def _try_rank54_algebraic():
    """
    Try to find rank-54 by attempting to absorb one schoolbook term into Strassen terms.
    
    Key idea: The schoolbook terms for A[i,2]*B[2,k] contribute to C[i,k].
    If we can modify the Strassen decomposition of the 2x2 block to also partially
    account for the 2x1 block contribution, we might save one term.
    
    Strategy: For each pair (i,k) in the 2x10 output, check if the schoolbook term
    A[i,2]*B[2,k] can be expressed as a linear combination of existing Strassen terms
    plus a modified term. This is a linear algebra problem.
    """
    n, m, p = DIMENSIONS
    base = _construct_rank55()
    U0, V0, W0 = base['U'], base['V'], base['W']
    r55 = 55
    r54 = 54
    
    # The schoolbook terms are columns 35-54 (indices 35 to 54)
    # They correspond to: for i in {0,1}, k in {0..9}: A[i,2]*B[2,k] -> C[i,k]
    # Let's try to see if we can eliminate one schoolbook term by modifying others
    
    # For a rank-54 decomposition, we need to find U,V,W of shape (6,54),(30,54),(20,54)
    # such that the tensor decomposition is exact.
    
    # Approach: Try to find a 54-term decomposition by solving a system of equations.
    # Fix U and V to have a specific sparse structure, then solve for W.
    
    # Let's try: keep the Strassen structure for the 2x2 part but modify it,
    # and use 19 terms for the 2x1 part instead of 20.
    
    # The 2x1 * 1x10 multiplication has rank at least 20 (it's a sum of 20 rank-1 tensors
    # with no algebraic dependencies in general). So we can't reduce that part alone.
    
    # Alternative: Use a different splitting. What if we split B into 3+7 or 4+6?
    # Or use a non-trivial bilinear algorithm for 2x3 * 3x10 directly?
    
    # Let's try a more direct approach: set up the linear system for rank 54.
    # We need to find U (6x54), V (30x54), W (20x54) such that:
    # For all (i,j,k) with i in {0,1}, j in {0,1,2}, k in {0..9}:
    #   sum_c U[i*3+j,c] * V[j*10+k,c] * W[k*2+i,c] = delta(i,j,k valid)
    
    # This is a system of 6*30*20 = 3600 equations in 54*54*54 = 157464 unknowns.
    # But it's nonlinear. However, we can use the fact that the solution space is large.
    
    # Better approach: Use the known rank-55 solution and try to find a linear dependency
    # among the 55 rank-1 tensors that allows us to express one as a combination of others.
    
    # Build the tensor matrix A where A[flat_idx, r] = U[i,r]*V[j,r]*W[k,r]
    I, J, K = n*m, m*p, p*n  # 6, 30, 20
    dim = I * J * K  # 3600
    
    A = np.zeros((dim, r55), dtype=float)
    for r in range(r55):
        term = np.einsum('i,j,k->ijk', U0[:,r], V0[:,r], W0[:,r])
        A[:,r] = term.flatten()
    
    # Check if the 55 columns are linearly dependent
    try:
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
        tol = 1e-10
        num_zeros = np.sum(s < tol)
        
        if num_zeros > 0:
            # We have a null space. Find a vector in the null space.
            null_vec = Vt[-1, :]  # Last right singular vector
            max_idx = np.argmax(np.abs(null_vec))
            if abs(null_vec[max_idx]) > 0.01:
                # Try to eliminate term max_idx
                # This requires solving a nonlinear system, which is hard.
                # For now, just report.
                pass
    except Exception:
        pass
    
    return None

def _try_rank54_direct():
    """
    Try to directly construct a rank-54 decomposition by solving a linear system
    with a fixed sparsity pattern.
    """
    n, m, p = DIMENSIONS
    r = 54
    
    # Fix U and V to be sparse with a specific pattern, then solve for W.
    # Use the Strassen pattern but with one fewer column.
    
    # Let's try: use 34 Strassen-like terms + 20 schoolbook terms = 54
    # This means we need to save one Strassen term.
    
    # The Strassen algorithm for 2x2 uses 7 terms. Can we do 2x2 in 6 terms? No,
    # it's proven that rank(2x2) = 7. But we're not doing pure 2x2, we're doing
    # 2x3 * 3x10, so the structure is different.
    
    # Let's try a different approach: use the fact that n=2 and p=10.
    # C = A*B where A is 2x3, B is 3x10.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # This is equivalent to: for each k, compute [C[0,k], C[1,k]]^T = A * B[:,k]
    # where A is 2x3 and B[:,k] is 3x1.
    
    # So we need to compute 10 independent 2x3 * 3x1 multiplications.
    # Each 2x3 * 3x1 has rank at most 6 (schoolbook), but can it be done in fewer?
    
    # Actually, the tensor for 2x3 * 3x1 is the same as the "matrix-vector" multiplication
    # tensor. Its rank is known to be 6 for generic matrices, but for this specific
    # structure, it might be lower.
    
    # Let's just return None and use the rank-55 solution.
    return None

def construct():
    if _time_remaining() < 5:
        result = _construct_rank55()
        return {'U': result['U'].tolist(), 'V': result['V'].tolist(), 'W': result['W'].tolist()}
    
    # Try to find rank 54
    better = _try_rank54_algebraic()
    if better is not None:
        if _verify_fast(np.array(better['U']), np.array(better['V']), np.array(better['W']), 2, 3, 10):
            return better
    
    better2 = _try_rank54_direct()
    if better2 is not None:
        if _verify_fast(np.array(better2['U']), np.array(better2['V']), np.array(better2['W']), 2, 3, 10):
            return better2
    
    # Return rank-55
    result = _construct_rank55()
    return {'U': result['U'].tolist(), 'V': result['V'].tolist(), 'W': result['W'].tolist()}
