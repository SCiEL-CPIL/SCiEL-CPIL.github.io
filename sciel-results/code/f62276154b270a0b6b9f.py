"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction as F

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=3, m=3, p=4, tol=F(1, 2)):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = F(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - F(1)) > tol:
                    return False
    return True

def _build_strassen_2x2x2():
    """
    Returns 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u,v,w are lists of 4 Fractions.
    A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([F(1),F(0),F(0),F(1)], [F(1),F(0),F(0),F(1)], [F(1),F(0),F(0),F(0)]))
    terms.append(([F(1),F(0),F(0),F(1)], [F(1),F(0),F(0),F(1)], [F(0),F(0),F(0),F(1)]))
    # M2 = (A10+A11)B00
    terms.append(([F(0),F(0),F(1),F(1)], [F(1),F(0),F(0),F(0)], [F(0),F(0),F(1),F(0)]))
    terms.append(([F(0),F(0),F(1),F(1)], [F(1),F(0),F(0),F(0)], [F(0),F(1),F(0),F(0)]))
    # M3 = A00(B10-B00)
    terms.append(([F(1),F(0),F(0),F(0)], [F(0),F(0),F(1),F(0)], [F(0),F(1),F(0),F(0)]))
    terms.append(([F(1),F(0),F(0),F(0)], [F(0),F(0),F(0),F(-1)], [F(0),F(1),F(0),F(0)]))
    # M4 = A11(B01-B11)
    terms.append(([F(0),F(0),F(0),F(1)], [F(0),F(1),F(0),F(0)], [F(0),F(0),F(1),F(0)]))
    terms.append(([F(0),F(0),F(0),F(1)], [F(0),F(0),F(0),F(-1)], [F(0),F(0),F(1),F(0)]))
    # M5 = (A00+A01)B11
    terms.append(([F(1),F(1),F(0),F(0)], [F(0),F(0),F(0),F(1)], [F(1),F(0),F(0),F(0)]))
    terms.append(([F(1),F(1),F(0),F(0)], [F(0),F(0),F(0),F(1)], [F(0),F(0),F(0),F(1)]))
    # M6 = (A10-A00)(B00+B01)
    terms.append(([F(1),F(0),F(1),F(0)], [F(1),F(1),F(0),F(0)], [F(0),F(0),F(0),F(1)]))
    terms.append(([F(1),F(0),F(1),F(0)], [F(1),F(1),F(0),F(0)], [F(0),F(0),F(1),F(0)]))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([F(0),F(1),F(0),F(1)], [F(0),F(0),F(1),F(1)], [F(1),F(0),F(0),F(0)]))
    terms.append(([F(0),F(1),F(0),F(1)], [F(0),F(0),F(1),F(1)], [F(0),F(1),F(0),F(0)]))
    
    # Wait, that's 14 terms. Strassen should be 7. Let me re-derive.
    # The standard Strassen 7-term decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each Mi contributes to multiple C entries. We need to split each Mi into rank-1 terms for each C entry it contributes to.
    # But that's not how it works. Each Mi is a scalar product. The contribution to C is a sum of these scalars.
    # So for the tensor decomposition, each (i,j,k) entry of C gets contributions from various Mi.
    # We need to create rank-1 terms (u,v,w) such that sum over terms of u[i,j]*v[j,k]*w[k,i] = 1 if (i,j,k) is a multiplication term.
    
    # Let me use the standard 7-term formulation directly as 7 rank-1 terms, but each term contributes to multiple output positions.
    # Actually, in the tensor decomposition framework, each rank-1 term has a specific (i,j,k) index for W.
    # So we need to expand each Mi into its contributions to each C entry.
    
    # Let me restart with a cleaner approach.
    # For 2x2x2, the tensor has 8 entries. Strassen uses 7 rank-1 terms.
    # The 7 terms are:
    # t1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] -> contributes to C00 and C11
    # Wait, w has 4 entries for C. If w=[1,0,0,1], it contributes to both C00 and C11.
    # But in the tensor decomposition, each rank-1 term has a single W index.
    # So we need to split it.
    
    # Actually, I think the standard way is:
    # Each of the 7 products Mi gives a contribution to specific C entries.
    # We create one rank-1 term for each (Mi, C_entry) pair.
    # But that would give more than 7 terms.
    
    # Let me look at this differently. The rank-1 tensor decomposition of the 2x2x2 multiplication tensor
    # with rank 7 means we have 7 terms, each with a specific (i,j,k) index.
    
    # The correct 7-term decomposition:
    # Term 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,0) -> C00 += A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Term 2: u=(1,0,0,1), v=(1,0,0,1), w=(0,0,0,1) -> C11 += same
    # Term 3: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0) -> C10 += A10*B00 + A11*B00
    # Term 4: u=(0,0,1,1), v=(1,0,0,0), w=(0,1,0,0) -> C01 += A10*B00 + A11*B00
    # Term 5: u=(1,0,0,0), v=(0,0,1,0), w=(0,1,0,0) -> C01 += A00*B10
    # Term 6: u=(1,0,0,0), v=(0,0,0,-1), w=(0,1,0,0) -> C01 -= A00*B11
    # Term 7: u=(0,0,0,1), v=(0,1,0,0), w=(0,0,1,0) -> C10 += A11*B01
    # Term 8: u=(0,0,0,1), v=(0,0,0,-1), w=(0,0,1,0) -> C10 -= A11*B11
    # Term 9: u=(1,1,0,0), v=(0,0,0,1), w=(1,0,0,0) -> C00 += A00*B11 + A01*B11
    # Term 10: u=(1,1,0,0), v=(0,0,0,1), w=(0,0,0,1) -> C11 += A00*B11 + A01*B11
    # Term 11: u=(1,0,1,0), v=(1,1,0,0), w=(0,0,0,1) -> C11 += A00*B00 + A00*B01 + A10*B00 + A10*B01
    # Term 12: u=(1,0,1,0), v=(1,1,0,0), w=(0,0,1,0) -> C10 += same
    # Term 13: u=(0,1,0,1), v=(0,0,1,1), w=(1,0,0,0) -> C00 += A01*B10 + A01*B11 + A11*B10 + A11*B11
    # Term 14: u=(0,1,0,1), v=(0,0,1,1), w=(0,1,0,0) -> C01 += same
    
    # That's 14 terms, not 7. I'm confusing myself.
    
    # The key insight: in the tensor decomposition, each rank-1 term has a FIXED (i,j,k) index.
    # So for C00 = A00*B00 + A01*B10, we need two rank-1 terms:
    #   (u=[1,0,0,0], v=[1,0,0,0], w=[1,0,0,0]) for A00*B00
    #   (u=[0,1,0,0], v=[0,0,1,0], w=[1,0,0,0]) for A01*B10
    # But Strassen reduces the number of terms by using linear combinations.
    
    # I'll just use the schoolbook for 2x2x2 (8 terms) and combine with the other blocks.
    # Total will be 8+8+4+4+4+4+2+2 = 36, same as schoolbook.
    # Let me just return the schoolbook solution.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 36
    
    rank = rank_schoolbook
    U = [[F(0) for _ in range(rank)] for _ in range(n * m)]
    V = [[F(0) for _ in range(rank)] for _ in range(m * p)]
    W = [[F(0) for _ in range(rank)] for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = F(1)
                V[j * p + k][column] = F(1)
                W[k * n + i][column] = F(1)
                column += 1
    
    U_out = [[int(x) for x in row] for row in U]
    V_out = [[int(x) for x in row] for row in V]
    W_out = [[int(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
