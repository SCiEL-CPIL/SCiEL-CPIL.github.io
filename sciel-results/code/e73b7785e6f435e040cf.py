"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U @ diag @ V^T reconstructs the matrix multiplication tensor."""
    # Build the full multiplication tensor T[i,j,k] = delta (A[i,j]*B[j,k] contributes to C[i,k])
    # T is shape (n*m, m*p, p*n) where T[a_idx, b_idx, c_idx] = 1 if a_idx=(i,j), b_idx=(j,k), c_idx=(k*n+i)
    # Actually the tensor is: T[(i,j), (j,k), (k*n+i)] = 1 for valid i,j,k
    # Decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    U_f = np.asarray(U, dtype=float)
    V_f = np.asarray(V, dtype=float)
    W_f = np.asarray(W, dtype=float)
    
    # Reconstruct: T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum
    T_recon = np.einsum('ar,br,cr->abc', U_f, V_f, W_f)
    
    return np.allclose(T_recon, T, atol=1e-9)

def _try_rank16():
    """
    Construct a rank-16 decomposition.
    Strategy: Fix 7 Strassen-like terms for the 2x2x2 core (A[0:2,0:2]*B[0:2,0:2]->C[0:2,0:2]).
    Then solve for 9 remaining terms covering the boundary interactions.
    
    The 18 output terms C[i,k] each need contributions from all m=3 intermediate j values.
    Total "schoolbook" terms = n*m*p = 18.
    
    We partition into 7 (Strassen core) + 9 (boundary) = 16 terms.
    
    Core covers: i in {0,1}, j in {0,1}, k in {0,1} -> 8 schoolbook terms compressed to 7.
    Boundary: all terms involving j=2 or k=2 (with i in {0,1}).
    - j=2, k in {0,1}: 4 terms (i=0,1; k=0,1)
    - j in {0,1}, k=2: 4 terms (i=0,1; j=0,1)
    - j=2, k=2: 2 terms (i=0,1)
    Total boundary = 10 terms. But we only have 9 slots.
    
    So we need to save 1 more term in the boundary. Let's try to merge two boundary terms.
    For example, A02*B20 and A02*B21 can't easily be merged.
    But A00*B02 and A01*B12 contribute to C02 - they have different A and B indices.
    
    Alternative: Use a different Strassen variant that covers more of the boundary.
    
    Let me think differently. The 18 terms are:
    (i,j,k) for i in 0,1; j in 0,1,2; k in 0,1,2
    
    Group by which "block" they belong to:
    - Block (i in 0,1; j in 0,1; k in 0,1): 8 terms -> 7 via Strassen
    - Block (i in 0,1; j=2; k in 0,1): 4 terms
    - Block (i in 0,1; j in 0,1; k=2): 4 terms
    - Block (i in 0,1; j=2; k=2): 2 terms
    Total: 8+4+4+2 = 18
    
    If we use 7 for the first block, we need 9 for the remaining 10 terms.
    We need to merge 2 of those 10 into 1 term.
    
    Possible merges:
    - A02*B20 -> C00 and A12*B20 -> C10: same B index (B20), different A. Can't merge into 1 rank-1 term.
    - A02*B20 -> C00 and A02*B21 -> C01: same A index (A02), different B. Can't merge.
    - A00*B02 -> C02 and A10*B02 -> C12: same B index (B02). Can't merge.
    
    Hmm, none of these can be merged into a single rank-1 term because a rank-1 term
    U[:,r] * V[:,r] * W[:,r] picks specific A-indices, B-indices, and C-indices.
    
    For a rank-1 term to cover two schoolbook terms (i1,j1,k1) and (i2,j2,k2),
    we need: U has 1s at (i1,j1) and (i2,j2), V has 1s at (j1,k1) and (j2,k2),
    W has 1s at (k1*n+i1) and (k2*n+i2).
    But then the product would also create cross terms (i1,j2), (j2,k1), etc.
    Unless those cross terms cancel or are zero.
    
    Actually, for a rank-1 term, U[a,r]=1 for specific a values, V[b,r]=1 for specific b values.
    The contribution to T[a,b,c] is U[a,r]*V[b,r]*W[c,r].
    If U has 1s at a1 and a2, V has 1s at b1 and b2, then T gets contributions at
    (a1,b1), (a1,b2), (a2,b1), (a2,b2) - all four combinations.
    So a single rank-1 term with |U support| = u, |V support| = v creates u*v "cells" in the
    A-B plane. For it to exactly match schoolbook terms, we need those u*v cells to correspond
    to valid (i,j,k) triples.
    
    This is getting complex. Let me try a direct approach: set up the linear system.
    
    For rank 16, we have 16 unknown rank-1 terms. Each term is defined by:
    - A subset of A-indices (6 possible)
    - A subset of B-indices (9 possible)  
    - A subset of C-indices (6 possible)
    
    This is a combinatorial search. Let me try a more structured approach.
    
    Actually, let me just try to find ANY rank-16 decomposition by solving the system.
    The tensor T has 18 non-zero entries. We need T = sum_{r=1}^{16} u_r v_r w_r.
    
    Let me try: fix the 7 Strassen terms, then see if the remaining 10 boundary terms
    can be covered by 9 rank-1 terms with some clever grouping.
    """
    n, m, p = DIMENSIONS
    
    # Strassen core: 7 terms for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Using standard Strassen decomposition
    strassen_U = []
    strassen_V = []
    strassen_W = []
    
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    # A00=(0,0)->idx0, A11=(1,1)->idx4
    # B00=(0,0)->idx0, B11=(1,1)->idx4
    # C00: k=0,i=0->idx0; C11: k=1,i=1->idx3
    strassen_U.append([0, 4])
    strassen_V.append([0, 4])
    strassen_W.append([0, 3])
    
    # M2 = (A10+A11)*B00 -> C10, C11
    # A10=(1,0)->idx3, A11=(1,1)->idx4
    # B00=(0,0)->idx0
    # C10: k=0,i=1->idx1; C11: k=1,i=1->idx3
    strassen_U.append([3, 4])
    strassen_V.append([0])
    strassen_W.append([1, 3])
    # But M2 should subtract from C11: W[3] = -1
    
    # M3 = A00*(B01-B11) -> C01, C11
    # A00=(0,0)->idx0
    # B01=(0,1)->idx1, B11=(1,1)->idx4
    # C01: k=1,i=0->idx2; C11: k=1,i=1->idx3
    strassen_U.append([0])
    strassen_V.append([1, 4])
    strassen_W.append([2, 3])
    
    # M4 = A11*(B10-B00) -> C00, C10
    # A11=(1,1)->idx4
    # B10=(1,0)->idx3, B00=(0,0)->idx0
    # C00: k=0,i=0->idx0; C10: k=0,i=1->idx1
    strassen_U.append([4])
    strassen_V.append([3, 0])
    strassen_W.append([0, 1])
    
    # M5 = (A00+A01)*B11 -> C00, C01
    # A00=(0,0)->idx0, A01=(0,1)->idx1
    # B11=(1,1)->idx4
    # C00: k=0,i=0->idx0; C01: k=1,i=0->idx2
    strassen_U.append([0, 1])
    strassen_V.append([4])
    strassen_W.append([0, 2])
    
    # M6 = (A10-A00)*(B00+B01) -> C11
    # A10=(1,0)->idx3, A00=(0,0)->idx0
    # B00=(0,0)->idx0, B01=(0,1)->idx1
    # C11: k=1,i=1->idx3
    strassen_U.append([3, 0])
    strassen_V.append([0, 1])
    strassen_W.append([3])
    
    # M7 = (A01-A11)*(B10+B11) -> C00
    # A01=(0,1)->idx1, A11=(1,1)->idx4
    # B10=(1,0)->idx3, B11=(1,1)->idx4
    # C00: k=0,i=0->idx0
    strassen_U.append([1, 4])
    strassen_V.append([3, 4])
    strassen_W.append([0])
    
    # Now let's build the full U, V, W matrices for these 7 terms
    U = np.zeros((n*m, 16), dtype=float)
    V = np.zeros((m*p, 16), dtype=float)
    W = np.zeros((p*n, 16), dtype=float)
    
    # Assign coefficients for Strassen terms
    # M1: +1, +1 -> +1, +1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: +1, +1; +1; +1, -1
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: +1; +1, -1; +1, +1
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: +1; +1, -1; +1, +1
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: +1, +1; +1; -1, +1
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: +1, -1; +1, +1; +1
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: +1, -1; +1, +1; +1
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Verify Strassen core
    # The 7 terms should exactly reproduce A[0:2,0:2]*B[0:2,0:2]->C[0:2,0:2]
    # Let's check by computing the contribution to the tensor
    
    # Now for the 9 boundary terms, we need to cover:
    # (i,j,k) where j=2 or k=2, with i in {0,1}
    # Terms:
    # j=2, k=0: (0,2,0)->C00, (1,2,0)->C10
    # j=2, k=1: (0,2,1)->C01, (1,2,1)->C11
    # j=0, k=2: (0,0,2)->C02, (1,0,2)->C12
    # j=1, k=2: (0,1,2)->C02, (1,1,2)->C12
    # j=2, k=2: (0,2,2)->C02, (1,2,2)->C12
    # Total: 10 terms
    
    # We have 9 slots (indices 7-15). Need to merge 2 terms into 1.
    # Let's try to see if any two boundary terms can share a rank-1 term.
    
    # Option: Use one rank-1 term to cover (0,2,0) and (0,2,2) -> both have A02
    # U: A02 (idx 2), V: B20 (idx 6) and B22 (idx 8), W: C00 (idx 0) and C02 (idx 4)
    # But this would also create cross terms: (A02, B20, C02) and (A02, B22, C00)
    # These are NOT valid schoolbook terms, so they'd be spurious.
    # Unless we can cancel them with other terms.
    
    # This is getting complicated. Let me try a different approach:
    # Just set up the linear system for all 18 terms with 16 unknowns.
    # Each rank-1 term r has unknowns: which A-indices, B-indices, C-indices it touches.
    # This is a combinatorial problem.
    
    # Alternative: Try rank 17 but with a smarter structure, or accept rank 17.
    # For now, let's return the rank-17 solution which we know works.
    
    return None  # Will fall back to rank 17

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try rank 16 first
    result16 = _try_rank16()
    if result16 is not None:
        U, V, W = result16
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 17 (known correct solution)
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen core (indices 0-6)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Boundary terms (indices 7-16)
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
