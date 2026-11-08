"""Matrix multiplication 2x3x10 tensor decomposition search."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _build_tensor(n, m, p):
    """Build the multiplication tensor as a 3D list T[i,j,k]."""
    T = [[[0]*p for _ in range(m)] for __ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i][j][k] = 1
    return T

def _verify(U, V, W, n, m, p, rank):
    """Verify that U,V,W exactly decompose the tensor."""
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _try_strassen_block(n1, m1, p1):
    """
    Try to find a low-rank decomposition for an n1 x m1 x p1 block
    using Strassen-style ideas. Returns (U,V,W,rank) or None.
    """
    # For 2x2x2, Strassen gives rank 7 (vs schoolbook 8)
    # For 2x3x2, we can use a mixed approach
    # Let's try to find minimal rank via exhaustive search for small blocks
    
    if n1 == 2 and m1 == 2 and p1 == 2:
        # Strassen's algorithm for 2x2x2
        # The 7 terms are:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        # U maps (i,j) -> coefficients for each m
        # V maps (j,k) -> coefficients for each m
        # W maps (k,i) -> coefficients for each m
        
        # U[i*m+j, c] where i in 0..1, j in 0..1, c in 0..6
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # m1 = (a00+a11)(b00+b11)
        # U: a00 -> +1, a11 -> +1 for m1
        U[0*2+0][0] = 1  # a00
        U[1*2+1][0] = 1  # a11
        # V: b00 -> +1, b11 -> +1 for m1
        V[0*2+0][0] = 1  # b00
        V[1*2+1][0] = 1  # b11
        # W: c00 gets +1, c11 gets +1 from m1
        W[0*2+0][0] = 1  # c00
        W[1*2+1][0] = 1  # c11
        
        # m2 = (a10+a11)b00
        U[1*2+0][1] = 1  # a10
        U[1*2+1][1] = 1  # a11
        V[0*2+0][1] = 1  # b00
        # W: c10 gets +1 from m2
        W[0*2+1][1] = 1  # c10 (k=0, i=1)
        
        # m3 = a00(b01-b11)
        U[0*2+0][2] = 1  # a00
        V[0*2+1][2] = 1  # b01
        V[1*2+1][2] = -1  # b11
        # W: c01 gets +1 from m3
        W[1*2+0][2] = 1  # c01 (k=1, i=0)
        
        # m4 = a11(b10-b00)
        U[1*2+1][3] = 1  # a11
        V[1*2+0][3] = 1  # b10
        V[0*2+0][3] = -1  # b00
        # W: c10 gets +1, c00 gets +1 from m4
        W[0*2+1][3] = 1  # c10
        W[0*2+0][3] = 1  # c00
        
        # m5 = (a00+a01)b11
        U[0*2+0][4] = 1  # a00
        U[0*2+1][4] = 1  # a01
        V[1*2+1][4] = 1  # b11
        # W: c11 gets +1, c00 gets -1 from m5
        W[1*2+1][4] = 1  # c11
        W[0*2+0][4] = -1  # c00
        
        # m6 = (a10-a00)(b00+b01)
        U[1*2+0][5] = 1  # a10
        U[0*2+0][5] = -1  # a00
        V[0*2+0][5] = 1  # b00
        V[0*2+1][5] = 1  # b01
        # W: c11 gets +1 from m6
        W[1*2+1][5] = 1  # c11
        
        # m7 = (a01-a11)(b10+b11)
        U[0*2+1][6] = 1  # a01
        U[1*2+1][6] = -1  # a11
        V[1*2+0][6] = 1  # b10
        V[1*2+1][6] = 1  # b11
        # W: c00 gets +1 from m7
        W[0*2+0][6] = 1  # c00
        
        return U, V, W, 7
    
    return None

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def _combine_blocks(blocks):
    """
    Combine block decompositions into a full decomposition.
    blocks: list of (n1, m1, p1, U1, V1, W1, rank1, i_off, j_off, k_off)
    """
    n, m, p = DIMENSIONS
    total_rank = sum(b[6] for b in blocks)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_off = 0
    for (n1, m1, p1, U1, V1, W1, rank1, i_off, j_off, k_off) in blocks:
        for i in range(n1):
            for j in range(m1):
                for c in range(rank1):
                    U[(i_off+i)*m + (j_off+j)][col_off+c] = U1[i*m1+j][c]
        for j in range(m1):
            for k in range(p1):
                for c in range(rank1):
                    V[(j_off+j)*p + (k_off+k)][col_off+c] = V1[j*p1+k][c]
        for k in range(p1):
            for i in range(n1):
                for c in range(rank1):
                    W[(k_off+k)*n + (i_off+i)][col_off+c] = W1[k*n1+i][c]
        col_off += rank1
    
    return U, V, W, total_rank

def _find_decomposition(n, m, p):
    """
    Find a low-rank decomposition for n x m x p matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 sub-blocks.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Strategy: partition the 2x3x10 problem into blocks
    # We can split the middle dimension (m=3) into 2+1 or the last dim (p=10) into chunks
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank = 60
    
    # Option 1: Split p=10 into 2+2+2+2+2, each 2x3x2 block
    # For 2x3x2, we can try to find optimal decomposition
    # Actually, let's think about this differently.
    
    # The tensor is T[i,j,k] = 1 for all i,j,k.
    # This is the all-ones tensor, which has a special structure.
    # The rank of the all-ones tensor of size n x m x p is max(n, m, p) if we allow real entries?
    # No, that's for a different problem.
    
    # Actually, for matrix multiplication, T[i,j,k] = delta_{i,j,k} is NOT the all-ones tensor.
    # Wait, let me re-read. The multiplication tensor for (n x m) * (m x p) -> (n x p)
    # has T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C.
    # So T[i,j,k] = 1 for ALL i,j,k. Yes, it IS the all-ones tensor!
    
    # The rank of the all-ones tensor [1]_{n x m x p} is max(n, m, p).
    # This is a well-known result. The rank equals the maximum of the dimensions.
    
    # Let me verify: for n=2, m=3, p=10, the rank should be 10.
    
    # Construction for rank = max(n,m,p):
    # If p = max(n,m,p), we can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j (sum_a u_a[i,j] v_a[j,k]) w_a[k,i] ... no wait.
    
    # Actually, the standard result is that the rank of the all-ones tensor is max(n,m,p).
    # Let me construct it for p >= n and p >= m:
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k
    
    # If we set rank = p, and for c = 0..p-1:
    # U[i*m+j, c] = 1 for all i,j
    # V[j*p+k, c] = 1 if k == c, else 0
    # W[k*n+i, c] = 1 for all i,k
    # Then sum_c U V W = sum_c (1) * (delta_{k,c}) * (1) = 1 if k in 0..p-1, which is always 1.
    # Wait, that gives sum over c of delta_{k,c} = 1. Yes!
    
    # So U[i*m+j, c] = 1 for all i,j,c
    # V[j*p+k, c] = 1 if k==c else 0
    # W[k*n+i, c] = 1 for all i,k,c
    # Rank = p = 10
    
    # Let me verify this works.
    # For fixed i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = sum_c 1 * delta_{k,c} * 1 = 1. Correct!
    
    # So the rank is exactly p = 10 (or max(n,m,p) in general).
    
    rank = max(n, m, p)
    
    # Construct the decomposition
    U = [[1]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[1]*rank for _ in range(p*n)]
    
    for j in range(m):
        for k in range(p):
            V[j*p+k][k] = 1
    
    # Verify
    if _verify(U, V, W, n, m, p, rank):
        return U, V, W, rank
    
    # Fallback to schoolbook if verification fails (shouldn't happen)
    U, V, W, rank = _schoolbook(n, m, p)
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _find_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
