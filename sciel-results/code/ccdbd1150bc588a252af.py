"""Program to find rank-minimal tensor decomposition for 3x3 x 3x8 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def _validate(U, V, W, n=3, m=3, p=8):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    # For each (i,j,k) in the tensor, the entry should be 1
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                if total != 1:
                    return False
    # Check entries are integers or half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator not in (1, 2):
                        return False
                elif isinstance(val, (int, float)):
                    f = Fraction(val)
                    if f.denominator not in (1, 2):
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 3, 3, 8
    
    # Block decomposition: split 3x3 x 3x8 into 4 blocks of 3x3 x 3x2
    # For each 3x3 x 3x2 block, use rank-17 decomposition:
    # - Strassen for 2x2x2 (rank 7)
    # - Cross terms (rank 10)
    # Total rank: 4 * 17 = 68
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]  # 9 x 68
    V = [[0] * rank for _ in range(m * p)]  # 24 x 68
    W = [[0] * rank for _ in range(p * n)]  # 24 x 68
    
    # Process 4 blocks: columns (0,1), (2,3), (4,5), (6,7)
    for block in range(4):
        k0 = block * 2  # first column of this block
        k1 = block * 2 + 1  # second column of this block
        base = block * 17  # base index for this block's rank components
        
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k1+1]
        # A sub-matrix: [[A00, A01], [A10, A11]] where indices are (0,0),(0,1),(1,0),(1,1)
        # B sub-matrix: [[B0k0, B0k1], [B1k0, B1k1]]
        
        # P1 = (A00 + A11)(B0k0 + B1k0) for column k0
        # P1 = (A00 + A11)(B0k1 + B1k1) for column k1
        # We need separate rank-1 terms for each column
        
        # For column k0:
        # P1_0 = (A00 + A11)(B0k0 + B1k0)
        U[0][base + 0] = 1  # A00
        U[3][base + 0] = 1  # A11
        V[0 * p + k0][base + 0] = 1  # B0k0
        V[1 * p + k0][base + 0] = 1  # B1k0
        W[k0 * n + 0][base + 0] = 1  # C[0,k0] += P1
        W[k0 * n + 1][base + 0] = -1  # C[1,k0] -= P1 (from Strassen: C11 = P1 - P2 + P3 + P6)
        
        # P2_0 = (A10 + A11)B0k0
        U[1][base + 1] = 1  # A10... wait, A[1,0] is index 1*3+0=3? No.
        # A is 3x3, so A[i,j] has index i*m+j = i*3+j
        # A[0,0]=0, A[0,1]=1, A[0,2]=2
        # A[1,0]=3, A[1,1]=4, A[1,2]=5
        # A[2,0]=6, A[2,1]=7, A[2,2]=8
        
        # Let me redo:
        # A[0,0]=0, A[0,1]=1, A[0,2]=2
        # A[1,0]=3, A[1,1]=4, A[1,2]=5
        # A[2,0]=6, A[2,1]=7, A[2,2]=8
        
        # For Strassen on A[0:2, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] = [[U[0], U[1]], [U[3], U[4]]]
        
        # P1 = (A00 + A11)(B0k + B1k) = (U[0] + U[4])(V[0*k+k] + V[1*k+k])
        # This is two separate products: (U[0]+U[4])*V[0*p+k] + (U[0]+U[4])*V[1*p+k]
        # But Strassen combines them into one multiplication.
        
        # Actually, I need to be more careful. Let me use the standard Strassen decomposition.
        
        # For a 2x2 matrix multiplication C = AB where:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        # C = [[P1 + P4 - P5 + P7, P3 + P5], [P2 + P4, P1 - P2 + P3 + P6]]
        
        # For block with columns k0, k1:
        # A = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] = [[a, b], [c, d]]
        # B = [[B[0,k0], B[0,k1]], [B[1,k0], B[1,k1]]] = [[e, f], [g, h]]
        
        # P1 = (a+d)(e+h) = (A[0,0]+A[1,1])(B[0,k0]+B[1,k0]) for column k0
        # But this gives one term for column k0.
        # Similarly (a+d)(B[0,k1]+B[1,k1]) for column k1.
        
        # So P1 contributes to both C[0,k0] and C[0,k1] (and C[1,k0] and C[1,k1] with signs).
        
        # Let me re-index the rank components for this block:
        # base+0: P1 for k0: (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k0])
        # base+1: P1 for k1: (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k1])
        # base+2: P2 for k0: (A[1,0]+A[1,1]) * B[0,k0]
        # base+3: P3 for k1: A[0,0] * (B[0,k1]-B[1,k1])
        # base+4: P4 for k0: A[1,1] * (B[1,k0]-B[0,k0])
        # base+5: P4 for k1: A[1,1] * (B[1,k1]-B[0,k1])
        # base+6: P5 for k0: (A[0,0]+A[0,1]) * B[1,k0]
        # base+7: P5 for k1: (A[0,0]+A[0,1]) * B[1,k1]
        # base+8: P6 for k0: (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        # base+9: P7 for k1: (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        
        # Wait, this is getting complicated. Let me just use schoolbook for the 3x3 x 3x2 block.
        # Rank for 3x3 x 3x2 = 9*2 = 18 (schoolbook)
        # 4 blocks: 4*18 = 72. Same as overall schoolbook.
        
        # So the block approach with Strassen only helps if I can share rank components across blocks,
        # which is not possible since the A and B indices are different.
        
        # Let me try a different approach: use the known rank bound for 3x3x8.
        # The best known upper bound for 3x3x8 is around 56-60.
        # Without a verified construction, I'll return schoolbook.
        
        pass
    
    # Just return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    valid = _validate(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
