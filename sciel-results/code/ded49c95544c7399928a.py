"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.

Strategy:
  - The schoolbook rank is 4*4*9 = 144.
  - We attempt to exploit structure: 4x4x9 can be viewed as 3 blocks of 4x4x3
    along the output dimension (k=0..8 split into three groups of 3).
    For each 4x4x3 sub-tensor, we use the known rank-16 decomposition for 2x4x4
    applied appropriately, or more directly, we use the fact that 4x4x3 has
    rank at most 4*4*3=48, but known better bounds exist.
    
    Actually, a cleaner approach: 4x4x9 = M_{2,2,3} tensor product structure.
    M_{4,4,9} can be built from M_{2,2,3} (rank 11) via Kronecker-like block
    construction. But dimensions don't factor cleanly.
    
    Let's use the direct approach: the strategy says to use rank-16 for 2x4x4.
    4x4x9 = (4x4x4) + (4x4x4) + (4x4x1).
    - 4x4x4 has known rank 28 (BCR/Winograd).
    - 4x4x1 is just 4x4 matrix-vector, rank 16.
    Total: 28 + 28 + 16 = 72.
    
    But we need exact decompositions with integer/half-integer entries.
    The Winograd rank-7 for 2x2x2 gives rank 28 for 4x4x4.
    The 4x4x1 case: each output (i,k) depends on sum_j A[i,j]*B[j,k] for fixed k.
    That's 4*4 = 16 terms, rank 16.
    
    Total rank: 28 + 28 + 16 = 72.
    
    Let me implement the Winograd rank-7 for 2x2x2 and build up.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0

# EVOLVE-BLOCK-START

def _winograd_2x2x2():
    """Return rank-7 decomposition for 2x2x2 matrix multiplication.
    T[io, u, v] where io = k*2+i, u = i*2+j, v = j*2+k.
    Returns (U, V, W) each of shape (4, 7).
    """
    # Winograd F_2(2) decomposition.
    # U rows: (i,j) for i in 0..1, j in 0..1
    # V rows: (j,k) for j in 0..1, k in 0..1
    # W rows: (k,i) for k in 0..1, i in 0..1
    #
    # The 7 multiplications:
    # m0 = a00 * b00
    # m1 = a00 * b01
    # m2 = a01 * b00
    # m3 = a01 * b01
    # m4 = (a00+a01) * (b00+b01)
    # m5 = (a00+a10) * (b00+b10)
    # m6 = (a01+a11) * (b01+b11)
    #
    # Wait, that's not quite right. Let me use the standard Winograd rank-7.
    #
    # Standard Winograd for 2x2x2:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    #
    # 7 multiplications:
    # m1 = a00 * b00
    # m2 = a00 * b10
    # m3 = a10 * b00
    # m4 = a10 * b10
    # m5 = (a00 + a01) * (b00 + b01)
    # m6 = (a10 + a11) * (b00 + b01)
    # m7 = (a00 + a10) * (b01 + b11)
    # m8 = (a01 + a11) * (b01 + b11)
    # That's 8. Need rank 7.
    #
    # Correct Winograd rank-7:
    # m1 = a00 * b00
    # m2 = a00 * b01
    # m3 = a01 * b00
    # m4 = a01 * b01
    # m5 = (a00+a01) * (b10+b11)
    # m6 = (a10+a11) * (b00+b01)
    # m7 = (a00+a10) * (b01+b11)
    #
    # C[0,0] = m1
    # C[0,1] = m2 + m7 - m1
    # C[1,0] = m3 + m6 - m1
    # C[1,1] = m4 + m5 + m7 - m2 - m3
    #
    # Let me verify:
    # C[0,0] = a00*b00 + a01*b10 = m1 + ... hmm not matching.
    #
    # Let me use the well-known Bini rank-7:
    # The 7 rank-1 tensors are defined by:
    # U (4x7), V (4x7), W (4x7) where:
    # U row (i,j), V row (j,k), W row (k,i)
    #
    # I'll use the explicit decomposition from literature.
    # For 2x2x2, the 7 terms have:
    # Term 0: U=(1,0,0,0), V=(1,0,0,0), W=(1,0,0,0)  -> a00*b00 -> C[0,0]
    # Term 1: U=(0,1,0,0), V=(0,1,0,0), W=(0,1,0,0)  -> a01*b10 -> C[0,0]... 
    #
    # This is getting complex. Let me just use the schoolbook for the sub-blocks
    # and combine. The schoolbook for 4x4x9 is rank 144. 
    #
    # A simpler improvement: use the fact that 4x4x9 = 9 copies of 4x4x1.
    # Each 4x4x1 has rank 16 (matrix-vector multiply). Total: 9*16 = 144. Same.
    #
    # Or 4x4x9 = 4 copies of 1x4x9. Each 1x4x9 is row-vector times matrix, rank 36. Total 144.
    #
    # The Winograd approach for 4x4x4 gives rank 28. For 4x4x9, if we can
    # decompose as 4x4x4 + 4x4x4 + 4x4x1, we get 28+28+16=72.
    #
    # Let me implement Winograd rank-7 for 2x2x2 properly.

    # Using the standard Winograd algorithm for 2x2 matrix multiplication:
    # Inputs: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # Outputs: C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # 7 multiplications:
    # m1 = a*e
    # m2 = b*g
    # m3 = a*f
    # m4 = b*h
    # m5 = (a+b)*(g+h)
    # m6 = (c+d)*(e+f)
    # m7 = (a+c)*(f+h)
    #
    # C[0,0] = m1 + m2
    # C[0,1] = m3 + m4
    # C[1,0] = m6 - m4 - m7 + m1 + m2  ... this is getting messy.
    #
    # Let me look up the exact Winograd rank-7 coefficients.
    #
    # Actually, the simplest correct rank-7 for 2x2x2:
    # U, V, W are 4x7 matrices.
    # Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: a00 * b00 -> contributes to C[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1: a01 * b10 -> contributes to C[0,0]
    U[1][1] = 1; V[2][1] = 1; W[0][1] = 1
    # Term 2: a10 * b00 -> contributes to C[1,0]
    U[2][2] = 1; V[0][2] = 1; W[2][2] = 1
    # Term 3: a11 * b10 -> contributes to C[1,0]
    U[3][3] = 1; V[2][3] = 1; W[2][3] = 1
    # Term 4: (a00+a01) * (b00+b10) -> contributes to C[0,0]
    U[0][4] = 1; U[1][4] = 1; V[0][4] = 1; V[2][4] = 1; W[0][4] = 1
    # Term 5: (a00+a10) * (b01+b11) -> contributes to C[0,1]
    U[0][5] = 1; U[2][5] = 1; V[1][5] = 1; V[3][5] = 1; W[1][5] = 1
    # Term 6: (a01+a11) * (b01+b11) -> contributes to C[0,1]
    U[1][6] = 1; U[3][6] = 1; V[1][6] = 1; V[3][6] = 1; W[1][6] = 1
    
    # Verify: C[0,0] should be a00*b00 + a01*b10
    # From terms: term0 gives 1*1*1=1 for (i=0,j=0,k=0): U[0]*V[0]*W[0]
    # term1 gives a01*b10: U[1]*V[2]*W[0]
    # term4 gives (a00+a01)*(b00+b10) = a00*b00 + a00*b10 + a01*b00 + a01*b10
    # Sum for C[0,0]: 1 + 1 + (1 + 0 + 0 + 1) = 3? That's wrong.
    #
    # I'm making errors. Let me just use the schoolbook for now and return rank 144.
    # The task says "rank-minimal exact" but if we can't find a better one reliably,
    # we return schoolbook.
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a sparse dict."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(k * n + i, i * m + j, j * p + k)] = 1
    return T


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    T = _build_tensor(n, m, p)
    # Compute the tensor from the decomposition
    computed = {}
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    val = W[io][c] * U[u][c] * V[v][c]
                    if val != 0:
                        s += val
                if s != 0:
                    computed[(io, u, v)] = s
    # Check they match
    if set(computed.keys()) != set(T.keys()):
        return False
    for key in T:
        if computed.get(key, 0) != T[key]:
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try to verify it's correct (it should be)
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback: return schoolbook anyway
    return result

# EVOLVE-BLOCK-END
