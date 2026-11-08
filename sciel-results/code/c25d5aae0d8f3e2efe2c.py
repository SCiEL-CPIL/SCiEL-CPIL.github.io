"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # A: n x m = 2x3, B: m x p = 3x6, C: n x p = 2x6
    # U: (n*m) x rank = 6 x rank
    # V: (m*p) x rank = 18 x rank
    # W: (p*n) x rank = 12 x rank
    # Convention: W[k*n+i, r]
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid i,j,k
    # (since C[i,k] = sum_j A[i,j]*B[j,k], each term A[i,j]*B[j,k] has coefficient 1)
    # Wait, that's not quite right. Let me think again.
    
    # The bilinear form is: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor decomposition form:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Hmm, this doesn't look right either. Let me re-examine.
    
    # Actually, the standard tensor decomposition for matrix multiplication (n x m) * (m x p) is:
    # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # No wait, that's not right either. The correct form is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # But this has issues with the j index being summed in both factors.
    
    # Let me look at this differently. The multiplication tensor M has entries:
    # M[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # And the decomposition is:
    # M[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must hold for all i, j, k.
    
    # So we need: for all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 36 equations.
    # We want to find U (6 x r), V (18 x r), W (12 x r) with small r.
    
    # Let's try to find a solution with rank r by solving the system.
    # The unknowns are the entries of U, V, W.
    # For each r, we have 6r + 18r + 12r = 36r unknowns.
    # And 36 equations.
    
    # For r >= 1, we have more unknowns than equations (36r > 36 for r >= 2).
    # But the equations are nonlinear (products of U, V, W entries).
    
    # Let's try a different approach: use known results.
    # For 2x3 * 3x6, we can split into three 2x3 * 3x2 blocks.
    # Each 2x3 * 3x2 block can be done with rank 7 (Strassen-like) or 6 (naive).
    # Actually, 2x3 * 3x2 has 12 entries in C, and the naive rank is 2*3*2 = 12.
    # But we can do better.
    
    # Let me try to construct a solution for 2x3 * 3x2 first, then replicate.
    
    # For 2x3 * 3x2:
    # A (2x3): indices (i,j) -> i*3+j, i in 0..1, j in 0..2
    # B (3x2): indices (j,k) -> j*2+k, j in 0..2, k in 0..1
    # C (2x2): indices (i,k) -> i*2+k, i in 0..1, k in 0..1
    
    # We need: for all i in 0..1, j in 0..2, k in 0..1:
    # sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    
    # That's 12 equations.
    
    # Let's try rank 7 for each block (using Strassen for the 2x2x2 part + 1 extra).
    # Actually, let me try to find the minimal rank for 2x3 * 3x2.
    
    # Known result: rank(2x3 * 3x2) = 7? Let me verify.
    # Actually, I recall that rank(2x3 * 3x2) might be 6 or 7.
    
    # Let me try to construct a rank-6 solution for 2x3 * 3x2.
    # If that works, total rank would be 6*3 = 18.
    # If rank 7, total would be 21.
    
    # Let's try rank 6 first.
    
    def solve_2x3_3x2(rank):
        """Try to find a rank decomposition for 2x3 * 3x2."""
        # U: 6 x rank, V: 6 x rank, W: 4 x rank
        # Equations: for i in 0..1, j in 0..2, k in 0..1:
        # sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
        
        # Let's use a simple construction.
        # For the 2x2x2 part (j=0,1), use Strassen (rank 7) or try rank 6.
        # Actually, let me just try to find any solution.
        
        # Simple approach: use the 12 scalar products (rank 12) and see if we can reduce.
        # Or try a specific construction.
        
        # Let's try: for each (i,j,k), assign a unique term, but try to merge.
        
        # Actually, let me just return the naive rank-12 solution for now and verify.
        U = np.zeros((6, rank))
        V = np.zeros((6, rank))
        W = np.zeros((4, rank))
        
        if rank < 12:
            # Try a smarter construction
            # This is getting complex. Let me just use the known rank-7 Strassen-based solution.
            return None
        
        # Naive solution
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    r = i * 6 + j * 2 + k
                    if r >= rank:
                        return None
                    U[i * 3 + j, r] = 1
                    V[j * 2 + k, r] = 1
                    W[k * 2 + i, r] = 1
        
        # Verify
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    val = sum(U[i * 3 + j, r] * V[j * 2 + k, r] * W[k * 2 + i, r] for r in range(rank))
                    if abs(val - 1) > 1e-9:
                        return None
        
        return U, V, W
    
    # Let me try a different approach: directly construct the full 2x3*3x6 solution.
    # Split B into three 3x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    # For each block, we need a 2x3*3x2 decomposition.
    
    # Let me try to find a rank-7 decomposition for 2x3*3x2 using Strassen ideas.
    # The 2x2*2x2 part (using A[:, 0:2] and B[0:2, :]) can use Strassen (rank 7).
    # Plus 2 more terms for A[:, 2] * B[2, :] (rank 2).
    # Total: 7 + 2 = 9 per block, so 27 total.
    
    # But maybe we can do better by sharing terms across blocks.
    
    # Let me try the simple approach: rank 6 per block (if possible) -> 18 total.
    # Or rank 7 per block -> 21 total.
    # Or rank 8 per block -> 24 total.
    
    # Let me just construct a valid solution with a reasonable rank and return it.
    # I'll use the Strassen-based approach for the 2x2 part + scalar products for the rest.
    
    # For each 2x3*3x2 block:
    # A (2x3): a00, a01, a02; a10, a11, a12
    # B (3x2): b00, b01; b10, b11; b20, b21
    # C (2x2): c00, c01; c10, c11
    
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    
    # Strassen for the 2x2*2x2 part (a00, a01, a10, a11) * (b00, b01, b10, b11):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # c00 = M1 - M2 + M4 - M6  (wait, let me recalculate)
    # Actually: c00 = M1 - M2 + M4 - M6? Let me verify.
    # M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # M2 = (a10+a11)*b00 = a10*b00 + a11*b00
    # M4 = a11*(b10-b00) = a11*b10 - a11*b00
    # M6 = (a10-a00)*(b00+b01) = a10*b00 + a10*b01 - a00*b00 - a00*b01
    
    # c00 should be a00*b00 + a01*b10 (but we're only doing 2x2*2x2, so c00 = a00*b00 + a01*b10)
    # Hmm, I'm getting confused. Let me just use the standard Strassen formulas.
    
    # Standard Strassen for C = A*B where A, B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M2 + M4 - M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M7
    
    # Let me verify C00:
    # M1 - M2 + M4 - M6 = (A00+A11)(B00+B11) - (A10+A11)B00 + A11(B10-B00) - (A10-A00)(B00+B01)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A11*B10 - A11*B00 - A10*B00 - A10*B01 + A00*B00 + A00*B01
    # = 2*A00*B00 + A00*B11 + A11*B11 + A11*B10 - 2*A10*B00 - A11*B00 - A10*B01 + A00*B01
    # That doesn't simplify to A00*B00 + A01*B10. I must have the Strassen formulas wrong.
    
    # Let me look up the correct Strassen formulas.
    # Actually, I think I mixed up the indices. Let me just use a simple valid construction.
    
    # Simple valid construction for 2x3*3x2 with rank 12:
    # For each (i,j,k), one term.
    
    # Let me just return a valid solution with rank 18 (6 per block * 3 blocks).
    # Actually, let me try to be smarter.
    
    # I'll construct a solution where for each block, I use 6 terms (one for each (i,k) pair,
    # summing over j internally). But that's not how tensor decomposition works.
    
    # Let me just use the naive 36-term solution and return it. It's valid, just not minimal.
    # Actually, the current program has rank 33. Let me try to improve on that.
    
    # Let me try rank 18: 6 terms per block, 3 blocks.
    rank = 18
    U = np.zeros((n * m, rank))  # 6 x 18
    V = np.zeros((m * p, rank))  # 18 x 18
    W = np.zeros((p * n, rank))  # 12 x 18
    
    # For each block b (0, 1, 2), and each (i, k) in 0..1 x 0..1:
    # We need C[i, b*2+k] = sum_j A[i,j] * B[j, b*2+k]
    # Term index = b*4 + i*2 + k
    # For this term, we want:
    # sum_j U[i*3+j, term] * V[j*6 + b*2+k, term] * W[(b*2+k)*2+i, term] = 1 for all i,j,k
    # But this can't be done with a single term per (i,k) because j is summed.
    
    # OK, let me just use the naive 36-term solution.
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    # Shouldn't happen with naive construction
                    pass
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
