"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Should equal 1 if i,j,k form a valid multiplication term, else 0
    # Actually, the tensor for matrix mult is T[i,j,k] = 1 for all i,j,k
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # Reconstruct: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    # Strassen's 7 products for 2x2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1+M4-M5+M7
    # c12 = M3+M5
    # c21 = M2+M4
    # c22 = M1-M2+M3+M6
    
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # U indices: i*m+j where i=0,1 and j=0,1
    # V indices: j*p+k where j=0,1 and k=0,1
    # W indices: k*n+i where k=0,1 and i=0,1
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1.0  # a11
    U[3][0] = 1.0  # a22
    V[0][0] = 1.0  # b11
    V[3][0] = 1.0  # b22
    W[0][0] = 1.0  # c11 (k=0,i=0)
    W[3][0] = 1.0  # c22 (k=1,i=1)
    
    # M2 = (a21+a22)b11
    U[2][1] = 1.0  # a21
    U[3][1] = 1.0  # a22
    V[0][1] = 1.0  # b11
    W[2][1] = 1.0  # c21 (k=0,i=1)
    
    # M3 = a11(b12-b22)
    U[0][2] = 1.0  # a11
    V[1][2] = 1.0  # b12
    V[3][2] = -1.0 # b22
    W[1][2] = 1.0  # c12 (k=1,i=0)
    
    # M4 = a22(b21-b11)
    U[3][3] = 1.0  # a22
    V[2][3] = 1.0  # b21
    V[0][3] = -1.0 # b11
    W[0][3] = 1.0  # c11 (k=0,i=0)
    W[2][3] = 1.0  # c21 (k=0,i=1)
    
    # M5 = (a11+a12)b22
    U[0][4] = 1.0  # a11
    U[1][4] = 1.0  # a12
    V[3][4] = 1.0  # b22
    W[0][4] = -1.0 # c11 (k=0,i=0)
    W[1][4] = 1.0  # c12 (k=1,i=0)
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1.0  # a21
    U[0][5] = -1.0 # a11
    V[0][5] = 1.0  # b11
    V[1][5] = 1.0  # b12
    W[3][5] = 1.0  # c22 (k=1,i=1)
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1.0  # a12
    U[3][6] = -1.0 # a22
    V[2][6] = 1.0  # b21
    V[3][6] = 1.0  # b22
    W[0][6] = 1.0  # c11 (k=0,i=0)
    W[3][6] = 1.0  # c22 (k=1,i=1)
    
    return U, V, W

def _combine_decompositions(decs):
    """Combine multiple decompositions by concatenation."""
    total_rank = sum(len(d[0][0]) for d in decs)
    n = m = p = 2
    U = [[0.0] * total_rank for _ in range(n*m)]
    V = [[0.0] * total_rank for _ in range(m*p)]
    W = [[0.0] * total_rank for _ in range(p*n)]
    
    offset = 0
    for U_d, V_d, W_d in decs:
        r = len(U_d[0])
        for i in range(n*m):
            for rr in range(r):
                U[i][offset+rr] += U_d[i][rr]
        for i in range(m*p):
            for rr in range(r):
                V[i][offset+rr] += V_d[i][rr]
        for i in range(p*n):
            for rr in range(r):
                W[i][offset+rr] += W_d[i][rr]
        offset += r
    
    return U, V, W

def _build_2x4x7_decomposition():
    """Build a decomposition for 2x4x7 using block methods."""
    n, m, p = 2, 4, 7
    
    # Strategy: Split the 4x7 matrix B into blocks and use Strassen where possible.
    # A is 2x4, B is 4x7, C is 2x7.
    
    # Split B into: B1 (4x4) and B2 (4x3)
    # A@B = A@B1 + A@B2
    # A@B1 is 2x4 times 4x4 -> 2x4, rank at most 2*4*4=32 schoolbook
    # A@B2 is 2x4 times 4x3 -> 2x3, rank at most 2*4*3=24 schoolbook
    # Total: 56 schoolbook
    
    # Better: Use Strassen on the 4x4 part.
    # Split A into A1 (2x2) and A2 (2x2)
    # Split B1 into B1_top (2x4) and B1_bottom (2x4)
    # A@B1 = A1@B1_top + A2@B1_bottom
    # Each is 2x2 times 2x4 -> 2x4
    # M_{2,2,4} can be done in rank... let's compute.
    
    # For M_{2,2,p}, we can use the fact that it's equivalent to multiplying
    # a 2x2 matrix by a 2xp matrix.
    # One approach: treat each column of the 2xp matrix separately.
    # But that gives p * rank(M_{2,2,1}) = p * 2 = 2p.
    # For p=4: rank 8.
    # For p=7: rank 14.
    
    # Actually, M_{2,2,p} has rank at most 2p using schoolbook per column.
    # Can we do better? For p=2, rank is 7 (Strassen).
    # For general p, rank(M_{2,2,p}) <= p + ceil(3p/4) or similar.
    
    # Let's try a different approach: direct search for low-rank decomposition.
    
    # Use a greedy algorithm to find a low-rank decomposition.
    # Start with schoolbook and try to merge terms.
    
    # For now, let's construct a rank-40 decomposition using a combination approach.
    
    # Split the 7 columns into 3+2+2 or 4+3.
    # Let's try: B = [B_a (4x4), B_b (4x3)]
    # A@B_a: 2x4 * 4x4
    #   Split A = [A1, A2] (each 2x2), B_a = [B_a1; B_a2] (each 2x4)
    #   A@B_a = A1@B_a1 + A2@B_a2
    #   Each M_{2,2,4}: use rank 8 (schoolbook per column)
    #   Total for A@B_a: 16
    # A@B_b: 2x4 * 4x3
    #   Similarly: A1@B_b1 + A2@B_b2, each M_{2,2,3}
    #   M_{2,2,3}: rank 6 (schoolbook)
    #   Total for A@B_b: 12
    # Grand total: 28
    
    # But wait, this is just schoolbook split, which gives the same total.
    # The key is that M_{2,2,p} might have rank less than 2p.
    
    # Known results: rank(M_{2,2,2}) = 7 < 8
    # For M_{2,2,3}: can we do better than 6?
    # For M_{2,2,4}: can we do better than 8?
    
    # Let's implement a search for M_{2,2,p} decompositions.
    
    def _search_2x2x_p(p, max_rank, time_limit=10.0):
        """Search for a low-rank decomposition of M_{2,2,p}."""
        start = time.time()
        n = m = 2
        
        # Schoolbook decomposition
        schoolbook_rank = n * m * p
        U_sb = [[0.0] * schoolbook_rank for _ in range(n*m)]
        V_sb = [[0.0] * schoolbook_rank for _ in range(m*p)]
        W_sb = [[0.0] * schoolbook_rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1.0
                    V_sb[j*p+k][col] = 1.0
                    W_sb[k*n+i][col] = 1.0
                    col += 1
        
        best_rank = schoolbook_rank
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        
        # Try to find a lower rank decomposition using random search
        # This is a simplified approach; in practice, one would use more sophisticated methods.
        
        # For now, return schoolbook as a safe fallback.
        return best_rank, best_U, best_V, best_W
    
    # For this implementation, let's use a known construction.
    # rank(M_{2,2,2}) = 7 (Strassen)
    # rank(M_{2,2,3}) <= 6 (schoolbook, no known improvement)
    # rank(M_{2,2,4}) <= 8 (schoolbook, but can use Strassen on 2x2 blocks)
    
    # For M_{2,2,4}: split the 4 columns into two 2-column blocks.
    # Each 2-column block is M_{2,2,2}, rank 7.
    # Total: 14. But schoolbook is 8, so this is worse!
    # So for M_{2,2,4}, schoolbook rank 8 is better.
    
    # Hmm, so Strassen doesn't help for wide matrices in this way.
    
    # Let me try a different strategy: use the fact that n=2 is small.
    # C = A@B where A is 2x4, B is 4x7.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0..6
    
    # This is a bilinear map. We want to find the minimal number of
    # "multiplication gates" needed.
    
    # Lower bound: max(n*p, m*p, n*m) = max(14, 28, 8) = 28
    # So rank >= 28.
    
    # Upper bound: 56 (schoolbook)
    
    # Let's try to construct a rank-35 decomposition.
    
    # One approach: use the fact that we can share computations.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # For each k, we need to compute 2 outputs from 4x4 inputs.
    # This is like doing 7 separate 2x4x1 multiplications.
    # Each 2x4x1 has rank at most 8 (schoolbook).
    # Total: 56.
    
    # But we can share across k values if we use the same A coefficients.
    
    # Let me try a direct construction with rank 36.
    
    # Use the following idea:
    # Compute intermediate values that are linear combinations of A and B,
    # then use them to compute C.
    
    # For now, let's return a safe schoolbook decomposition with rank 56.
    # In a future iteration, we can search for better decompositions.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try to find a low-rank decomposition
    U, V, W = _build_2x4x7_decomposition()
    
    rank = len(U[0])
    
    # Verify the decomposition
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
    
    # Convert to lists if needed
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
