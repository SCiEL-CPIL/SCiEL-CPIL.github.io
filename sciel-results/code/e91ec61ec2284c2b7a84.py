import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    The decomposition satisfies: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Total rank calculation:
    # Part 1 (j=2): M_{3,1,7} -> rank 21
    # Part 2 (j in {0,1}, i=2): M_{1,2,7} -> rank 14
    # Part 3 (j in {0,1}, i in {0,1}): M_{2,2,7} -> rank 25 (3*7 Strassen + 4 Schoolbook)
    total_rank = 21 + 14 + 25
    rank = total_rank
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Helper to add a term to the decomposition
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # coeff: coefficient for this term in the current column c
    def add_term(c, u_idx, coeff_u, v_idx, coeff_v, w_idx, coeff_w):
        if u_idx is not None:
            U[u_idx, c] += coeff_u
        if v_idx is not None:
            V[v_idx, c] += coeff_v
        if w_idx is not None:
            W[w_idx, c] += coeff_w

    # Part 1: j = 2
    # Tensor M_{3,1,7}: A[i,2] * B[2,k] -> C[i,k]
    # This is a sum of 21 rank-1 tensors: A[i,2] B[2,k] W[k*n+i]
    for i in range(n):
        for k in range(p):
            add_term(col, i*m + 2, 1.0, 2*p + k, 1.0, k*n + i, 1.0)
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Part 2: j in {0,1}, i = 2
    # Tensor M_{1,2,7}: A[2,j] * B[j,k] -> C[2,k]
    # This is a sum of 14 rank-1 tensors: A[2,j] B[j,k] W[k*n+2]
    if col < rank:
        for j in range(2):
            for k in range(p):
                add_term(col, 2*m + j, 1.0, j*p + k, 1.0, k*n + 2, 1.0)
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

    # Part 3: j in {0,1}, i in {0,1}
    # Tensor M_{2,2,7}
    # Split k into {0,1}, {2,3}, {4,5} for Strassen, and {6} for Schoolbook
    
    def add_strassen_block(i0, k0, j0, c):
        """
        Adds a 7-term Strassen decomposition for M_{2,2,2}
        Global indices:
        i in {i0, i0+1}
        j in {j0, j0+1} (here j0 is always 0 for Part 3)
        k in {k0, k0+1}
        
        Local A indices:
        A00 -> A[i0, j0]      -> U[i0*m + j0]
        A01 -> A[i0, j0+1]    -> U[i0*m + (j0+1)]
        A10 -> A[i0+1, j0]    -> U[(i0+1)*m + j0]
        A11 -> A[i0+1, j0+1]  -> U[(i0+1)*m + (j0+1)]
        
        Local B indices:
        B00 -> B[j0, k0]      -> V[j0*p + k0]
        B01 -> B[j0, k0+1]    -> V[j0*p + (k0+1)]
        B10 -> B[j0+1, k0]    -> V[(j0+1)*p + k0]
        B11 -> B[j0+1, k0+1]  -> V[(j0+1)*p + (k0+1)]
        
        Local C indices:
        C00 -> C[i0, k0]      -> W[k0*n + i0]
        C01 -> C[i0, k0+1]    -> W[(k0+1)*n + i0]
        C10 -> C[i0+1, k0]    -> W[k0*n + (i0+1)]
        C11 -> C[i0+1, k0+1]  -> W[(k0+1)*n + (i0+1)]
        """
        
        # 1. M1 = A00 * (B01 - B11)
        # U: A00 (1)
        # V: B01 (1), B11 (-1)
        # W: C01 (1), C11 (1)  -- Wait, C01 corresponds to M1+M7. C11 corresponds to M2-M1+M3+M6.
        # Let's trace coefficients for W.
        # W[k*n+i] gets contribution from terms where C[i,k] has a non-zero coefficient.
        # For M1:
        #   C01 coeff is 1. C11 coeff is -1? No, M1 appears in C01 (+1) and C11 (-1).
        #   So W[(k0+1)*n + i0] += 1
        #   W[(k0+1)*n + (i0+1)] += -1
        add_term(c, i0*m + j0, 1.0, 
                 j0*p + (k0+1), 1.0, 
                 (k0+1)*n + i0, 1.0)
        add_term(c, None, 0.0,
                 (j0+1)*p + (k0+1), -1.0,
                 (k0+1)*n + (i0+1), -1.0)
        
        # 2. M2 = (A00 + A10) * B11
        # U: A00 (1), A10 (1)
        # V: B11 (1)
        # W: C00 coeff -1, C11 coeff 1
        #   So W[k0*n + i0] += -1
        #   W[(k0+1)*n + (i0+1)] += 1
        add_term(c, i0*m + j0, 1.0,
                 (j0+1)*p + (k0+1), 1.0,
                 k0*n + i0, -1.0)
        add_term(c, (i0+1)*m + j0, 1.0,
                 None, 0.0,
                 (k0+1)*n + (i0+1), 1.0)
        
        # 3. M3 = (A10 - A00) * B00
        # U: A10 (1), A00 (-1)
        # V: B00 (1)
        # W: C10 coeff 1, C11 coeff 1
        #   So W[k0*n + (i0+1)] += 1
        #   W[(k0+1)*n + (i0+1)] += 1
        add_term(c, (i0+1)*m + j0, 1.0,
                 j0*p + k0, 1.0,
                 k0*n + (i0+1), 1.0)
        add_term(c, i0*m + j0, -1.0,
                 None, 0.0,
                 (k0+1)*n + (i0+1), 1.0)
        
        # 4. M4 = A11 * (B00 - B10)
        # U: A11 (1)
        # V: B00 (1), B10 (-1)
        # W: C00 coeff 1, C10 coeff 1
        #   So W[k0*n + i0] += 1
        #   W[k0*n + (i0+1)] += 1
        add_term(c, (i0+1)*m + (j0+1), 1.0,
                 j0*p + k0, 1.0,
                 k0*n + i0, 1.0)
        add_term(c, None, 0.0,
                 (j0+1)*p + k0, -1.0,
                 k0*n + (i0+1), 1.0)
        
        # 5. M5 = (A00 + A11) * (B00 + B11)
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 coeff 1, C10 coeff 1, C01? No.
        # C00 = M5 + ... (+1)
        # C10 = M3 + M5 (+1)
        # So W[k0*n + i0] += 1
        #    W[k0*n + (i0+1)] += 1
        # Note: M5 does not appear in C01 or C11.
        add_term(c, i0*m + j0, 1.0,
                 j0*p + k0, 1.0,
                 k0*n + i0, 1.0)
        add_term(c, (i0+1)*m + (j0+1), 1.0,
                 (j0+1)*p + (k0+1), 1.0,
                 k0*n + (i0+1), 1.0)
        
        # 6. M6 = (A01 - A11) * (B10 + B11)
        # U: A01 (1), A11 (-1)
        # V: B10 (1), B11 (1)
        # W: C11 coeff 1
        #   So W[(k0+1)*n + (i0+1)] += 1
        add_term(c, i0*m + (j0+1), 1.0,
                 (j0+1)*p + k0, 1.0,
                 (k0+1)*n + (i0+1), 1.0)
        add_term(c, (i0+1)*m + (j0+1), -1.0,
                 (j0+1)*p + (k0+1), 1.0,
                 None, 0.0)
        
        # 7. M7 = (A00 + A01) * (B01 - B10)
        # U: A00 (1), A01 (1)
        # V: B01 (1), B10 (-1)
        # W: C01 coeff 1, C00 coeff -1
        #   So W[(k0+1)*n + i0] += 1
        #   W[k0*n + i0] += -1
        add_term(c, i0*m + j0, 1.0,
                 j0*p + (k0+1), 1.0,
                 (k0+1)*n + i0, 1.0)
        add_term(c, i0*m + (j0+1), 1.0,
                 (j0+1)*p + k0, -1.0,
                 k0*n + i0, -1.0)

    # Apply Strassen for k blocks {0,1}, {2,3}, {4,5}
    # i0 is always 0 for this part (i in {0,1})
    # j0 is always 0 for this part (j in {0,1})
    if col < rank:
        add_strassen_block(0, 0, 0, col)
        col += 1
        if col < rank:
            add_strassen_block(0, 2, 0, col)
            col += 1
        if col < rank:
            add_strassen_block(0, 4, 0, col)
            col += 1

    # Apply Schoolbook for k block {6}
    # i in {0,1}, j in {0,1}, k=6
    if col < rank:
        for i in range(2):
            for j in range(2):
                add_term(col, i*m + j, 1.0, j*p + 6, 1.0, 6*n + i, 1.0)
                col += 1
                if col >= rank:
                    break
            if col >= rank:
                break

    # Verification and cleanup
    # If we didn't fill all columns, or if col != rank, there's an error.
    # For a fixed decomposition, col should match rank exactly.
    if col != rank:
        # Fallback to schoolbook if structure failed
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        rank = c

    # Convert to lists with integer entries
    U_list = [[int(round(x)) for x in row] for row in U]
    V_list = [[int(round(x)) for x in row] for row in V]
    W_list = [[int(round(x)) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
