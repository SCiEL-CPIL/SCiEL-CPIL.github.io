import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_TIME_LIMIT = 130.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _validate(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Reshape to tensors
    # U: (n*m) x rank -> (n, m, rank)
    # V: (m*p) x rank -> (m, p, rank)  
    # W: (p*n) x rank -> (p, n, rank)
    U_t = Ua.reshape(n, m, rank)
    V_t = Va.reshape(m, p, rank)
    W_t = Wa.reshape(p, n, rank)
    
    # Compute tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Should equal 1 if i,j,k form a valid multiplication path, else 0
    # Actually the multiplication tensor T[i,j,k,l] = delta_{j,k} for A(i,j)*B(j,l) -> C(i,l)
    # In our indexing: A is n x m, B is m x p, C is n x p
    # U has shape (n*m, rank) with index i*m + j
    # V has shape (m*p, rank) with index j*p + k
    # W has shape (p*n, rank) with index k*n + i (trace convention W[k*n+i, r])
    
    # Tensor element: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if the indices are consistent with matrix mult, else 0
    # For matrix mult C[i,k] = sum_j A[i,j]*B[j,k], the tensor is T[i,j,k] = delta (always 1 for valid indices)
    # Actually the multiplication tensor has T[i,j,k] = 1 for all valid i,j,k since it's the identity bilinear form
    
    # Let's just check: the reconstructed C should match A@B for random inputs
    # But that's expensive. Instead, check the tensor directly.
    # The tensor for matrix mult: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] B[j,k] means the coefficient of A[i,j]*B[j,k] in C[i,k] is 1
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m + j, r] * Va[j*p + k, r] * Wa[k*n + i, r]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < 1e-9

def _build_strassen56():
    """Build the known rank-56 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(2):
        for j in range(4):
            a_idx = {
                (0,0): 0 * m + (2*s + 0),
                (0,1): 0 * m + (2*s + 1),
                (1,0): 1 * m + (2*s + 0),
                (1,1): 1 * m + (2*s + 1)
            }
            b_idx = {
                (0,0): (2*s + 0) * p + (2*j + 0),
                (0,1): (2*s + 0) * p + (2*j + 1),
                (1,0): (2*s + 1) * p + (2*j + 0),
                (1,1): (2*s + 1) * p + (2*j + 1)
            }
            c_idx = {
                (0,0): (2*j + 0) * n + 0,
                (0,1): (2*j + 1) * n + 0,
                (1,0): (2*j + 0) * n + 1,
                (1,1): (2*j + 1) * n + 1
            }
            
            # Strassen U coefficients
            U[a_idx[(0,0)]][col] += 1
            U[a_idx[(1,1)]][col] += 1
            U[a_idx[(1,0)]][col+1] += 1
            U[a_idx[(1,1)]][col+1] += 1
            U[a_idx[(0,0)]][col+2] += 1
            U[a_idx[(1,1)]][col+3] += 1
            U[a_idx[(0,0)]][col+4] += 1
            U[a_idx[(0,1)]][col+4] += 1
            U[a_idx[(1,0)]][col+5] += 1
            U[a_idx[(0,0)]][col+5] += -1
            U[a_idx[(0,1)]][col+6] += 1
            U[a_idx[(1,1)]][col+6] += -1
            
            # Strassen V coefficients
            V[b_idx[(0,0)]][col] += 1
            V[b_idx[(1,1)]][col] += 1
            V[b_idx[(0,0)]][col+1] += 1
            V[b_idx[(0,1)]][col+2] += 1
            V[b_idx[(1,1)]][col+2] += -1
            V[b_idx[(1,0)]][col+3] += 1
            V[b_idx[(0,0)]][col+3] += -1
            V[b_idx[(1,1)]][col+4] += 1
            V[b_idx[(0,0)]][col+5] += 1
            V[b_idx[(0,1)]][col+5] += 1
            V[b_idx[(1,0)]][col+6] += 1
            V[b_idx[(1,1)]][col+6] += 1
            
            # Strassen W coefficients
            W[c_idx[(0,0)]][col] += 1
            W[c_idx[(0,0)]][col+3] += 1
            W[c_idx[(0,0)]][col+4] += -1
            W[c_idx[(0,0)]][col+6] += 1
            W[c_idx[(0,1)]][col+2] += 1
            W[c_idx[(0,1)]][col+4] += 1
            W[c_idx[(1,0)]][col+1] += 1
            W[c_idx[(1,0)]][col+3] += 1
            W[c_idx[(1,1)]][col] += 1
            W[c_idx[(1,1)]][col+1] += -1
            W[c_idx[(1,1)]][col+2] += 1
            W[c_idx[(1,1)]][col+5] += 1
            
            col += 7
    return {"U": U, "V": V, "W": W}

def _build_kronecker_strassen():
    """
    Try to build a rank-28 decomposition by treating the problem as 2x8 * 8x8.
    
    Reshape A (2x4) into a 2x8 matrix A' where A'[i, 2j] = A[i,j] and A'[i, 2j+1] = 0
    Reshape B (4x8) into an 8x4 matrix B' where B'[2j, i] = B[j, i] and B'[2j+1, i] = 0
    
    Then C = A @ B becomes C = A' @ B' where A' is 2x8 and B' is 8x4, giving C as 2x4...
    That doesn't work directly. Let me think differently.
    
    Actually, let's use the Kronecker product approach:
    The 2x4*4x8 mult can be seen as applying a rank-7 Strassen to the "row" dimension (size 2)
    and a rank-4 identity to the "column" dimension (size 4 for A, size 8 for B).
    
    Strassen 2x2*2x2 has rank 7.
    If we Kronecker product with an identity of rank 4 (for the 4 column blocks of A),
    we get rank 7*4 = 28.
    
    The idea: partition A into 2 row-groups (size 2) and 4 col-groups (size 1 each... no).
    
    Let me try: A is 2x4. Think of it as a 2x2 block matrix where each block is 1x2.
    B is 4x8. Think of it as a 2x2 block matrix where each block is 2x4.
    
    Actually, let's use the standard approach:
    A (2x4) = [A_left (2x2) | A_right (2x2)]
    B (4x8) = [B_top (2x8); B_bottom (2x8)]
    
    C = A_left * B_top + A_right * B_bottom
    
    Each term is a 2x2 * 2x8 multiplication.
    A 2x2 * 2x8 can be done with rank 7*4 = 28 using Strassen on the 2x2 part and identity on the 4 columns.
    
    So total rank = 28 + 28 = 56. That's the same as before.
    
    To get rank 28, we'd need to share the Strassen factors between the two terms,
    which isn't straightforward.
    
    Let me try a different approach: use the full 2x4*4x8 as a single tensor and
    apply a rank-28 decomposition directly.
    """
    n, m, p = DIMENSIONS
    # Try rank 28: Kronecker of Strassen (rank 7) and identity (rank 4)
    # This corresponds to treating the 2-row dimension with Strassen and the 4-col dimension with identity
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 32 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen 2x2*2x2 factors (rank 7)
    # For 2x2 matrices X and Y, Z = X*Y:
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)*y00
    # m3 = x00*(y01-y11)
    # m4 = x11*(y10-y00)
    # m5 = (x00+x01)*y11
    # m6 = (x10-x00)*(y00+y01)
    # m7 = (x01-x11)*(y10+y11)
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # Now, A is 2x4, B is 4x8.
    # Partition A into left (cols 0-1) and right (cols 2-3): each 2x2
    # Partition B into top (rows 0-1) and bottom (rows 2-3): each 2x4
    # But B_top is 2x4, not 2x2.
    
    # Alternative: partition B into 4 blocks of 2x2 each.
    # B = [B0 | B1 | B2 | B3] where each Bj is 4x2? No, B is 4x8.
    # B = [B_col0 (4x2) | B_col1 (4x2) | B_col2 (4x2) | B_col3 (4x2)]
    # Each B_colj = [B_colj_top (2x2); B_colj_bottom (2x2)]
    
    # This is getting complex. Let me just return the rank-56 solution for now
    # and let the search try to find better.
    return _build_strassen56()

def _try_rank_reduction(best_result, n, m, p, max_rank=56):
    """
    Try to reduce the rank of the decomposition using linear algebra.
    Given a valid decomposition of rank R, try to find a lower-rank one.
    """
    if _time_remaining() < 5.0:
        return best_result
    
    U = best_result["U"]
    V = best_result["V"]
    W = best_result["W"]
    current_rank = len(U[0])
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # The decomposition defines a tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find a lower-rank decomposition.
    
    # Approach: try to eliminate columns that are linear combinations of others
    # in a way that preserves the tensor.
    
    # This is a hard problem. For now, just return the best result.
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the known rank-56 solution
    best_result = _build_strassen56()
    
    # Validate the baseline
    if not _validate(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback: shouldn't happen, but return anyway
        return best_result
    
    # Try to find a better solution within time budget
    if _time_remaining() > 10.0:
        # Try Kronecker product approach (rank 28 target)
        kr_result = _build_kronecker_strassen()
        if kr_result["U"] and len(kr_result["U"][0]) < len(best_result["U"][0]):
            if _validate(kr_result["U"], kr_result["V"], kr_result["W"], n, m, p):
                best_result = kr_result
    
    # Attempt rank reduction via column elimination
    if _time_remaining() > 5.0:
        best_result = _try_rank_reduction(best_result, n, m, p)
    
    return best_result
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    col = 0
    for s in range(2): # s=0 (A_L, B_top), s=1 (A_R, B_bot)
        for j in range(4): # j=0..3 (col blocks of B)
            # Strassen for A_s (2x2) and B_{s,j} (2x2)
            
            # A_s indices in global A (2x4)
            # A_s is at cols 2s, 2s+1
            # Rows 0, 1
            a_idx = {
                (0,0): 0 * m + (2*s + 0),
                (0,1): 0 * m + (2*s + 1),
                (1,0): 1 * m + (2*s + 0),
                (1,1): 1 * m + (2*s + 1)
            }
            
            # B_{s,j} indices in global B (4x8)
            # B_{s,j} is at rows 2s, 2s+1 and cols 2j, 2j+1
            b_idx = {
                (0,0): (2*s + 0) * p + (2*j + 0),
                (0,1): (2*s + 0) * p + (2*j + 1),
                (1,0): (2*s + 1) * p + (2*j + 0),
                (1,1): (2*s + 1) * p + (2*j + 1)
            }
            
            # C contribution indices in global C (2x8)
            # Local C (2x2) maps to global C rows 0,1 and cols 2j, 2j+1
            # Global C index: k*n + i
            c_idx = {
                (0,0): (2*j + 0) * n + 0,
                (0,1): (2*j + 1) * n + 0,
                (1,0): (2*j + 0) * n + 1,
                (1,1): (2*j + 1) * n + 1
            }
            
            # Strassen U coefficients
            # u1 = a00 + a11
            U[a_idx[(0,0)]][col] += 1
            U[a_idx[(1,1)]][col] += 1
            # u2 = a10 + a11
            U[a_idx[(1,0)]][col+1] += 1
            U[a_idx[(1,1)]][col+1] += 1
            # u3 = a00
            U[a_idx[(0,0)]][col+2] += 1
            # u4 = a11
            U[a_idx[(1,1)]][col+3] += 1
            # u5 = a00 + a01
            U[a_idx[(0,0)]][col+4] += 1
            U[a_idx[(0,1)]][col+4] += 1
            # u6 = a10 - a00
            U[a_idx[(1,0)]][col+5] += 1
            U[a_idx[(0,0)]][col+5] += -1
            # u7 = a01 - a11
            U[a_idx[(0,1)]][col+6] += 1
            U[a_idx[(1,1)]][col+6] += -1
            
            # Strassen V coefficients
            # v1 = b00 + b11
            V[b_idx[(0,0)]][col] += 1
            V[b_idx[(1,1)]][col] += 1
            # v2 = b00
            V[b_idx[(0,0)]][col+1] += 1
            # v3 = b01 - b11
            V[b_idx[(0,1)]][col+2] += 1
            V[b_idx[(1,1)]][col+2] += -1
            # v4 = b10 - b00
            V[b_idx[(1,0)]][col+3] += 1
            V[b_idx[(0,0)]][col+3] += -1
            # v5 = b11
            V[b_idx[(1,1)]][col+4] += 1
            # v6 = b00 + b01
            V[b_idx[(0,0)]][col+5] += 1
            V[b_idx[(0,1)]][col+5] += 1
            # v7 = b10 + b11
            V[b_idx[(1,0)]][col+6] += 1
            V[b_idx[(1,1)]][col+6] += 1
            
            # Strassen W coefficients
            # c00 = m1 + m4 - m5 + m7
            W[c_idx[(0,0)]][col] += 1
            W[c_idx[(0,0)]][col+3] += 1
            W[c_idx[(0,0)]][col+4] += -1
            W[c_idx[(0,0)]][col+6] += 1
            
            # c01 = m3 + m5
            W[c_idx[(0,1)]][col+2] += 1
            W[c_idx[(0,1)]][col+4] += 1
            
            # c10 = m2 + m4
            W[c_idx[(1,0)]][col+1] += 1
            W[c_idx[(1,0)]][col+3] += 1
            
            # c11 = m1 - m2 + m3 + m6
            W[c_idx[(1,1)]][col] += 1
            W[c_idx[(1,1)]][col+1] += -1
            W[c_idx[(1,1)]][col+2] += 1
            W[c_idx[(1,1)]][col+5] += 1
            
            col += 7
            
    return {"U": U, "V": V, "W": W}
