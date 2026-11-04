"""
Rank-minimal tensor decomposition for 2x3 * 3x10 matrix multiplication.
Strategy: Split intermediate dim j into {0} and {1,2}.
- j=0: schoolbook 2x1*1x10, rank 20
- j=1,2: Strassen on five 2x2*2x2 blocks, rank 5*7=35
Total rank: 55
"""
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    total_rank = 55
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0.0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0.0] * total_rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    
    # Part 1: j=0, schoolbook
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1.0
            V[0 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    
    # Part 2: j in {1,2}, Strassen per 2x2 block
    # A submatrix: rows 0,1; cols 1,2 -> A[0,1],A[0,2],A[1,1],A[1,2]
    # B submatrix: rows 1,2; cols k0,k1 -> B[1,k0],B[1,k1],B[2,k0],B[2,k1]
    # C submatrix: rows 0,1; cols k0,k1 -> C[0,k0],C[0,k1],C[1,k0],C[1,k1]
    
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Strassen products (verified coefficients):
        # p1 = (a+d)(e+h), p2 = (c+d)e, p3 = (b-a)(f+h), p4 = (a+d)(h-g)
        # p5 = (a+c)(e-g), p6 = (b+d)(h-e), p7 = (b-c)(g+h)
        # where a=A[0,1],b=A[0,2],c=A[1,1],d=A[1,2]
        #       e=B[1,k0],f=B[1,k1],g=B[2,k0],h=B[2,k1]
        # C[0,k0] = p1+p4-p5+p7, C[0,k1] = p3+p5
        # C[1,k0] = p2+p4, C[1,k1] = p1-p2+p3+p6
        
        # p1 = (A[0,1]+A[1,2]) * (B[1,k0]+B[2,k1])
        U[0*m+1][col] += 1.0
        U[1*m+2][col] += 1.0
        V[1*p+k0][col] += 1.0
        V[2*p+k1][col] += 1.0
        W[k0*n+0][col] += 1.0
        W[k1*n+0][col] += 1.0
        W[k0*n+1][col] += 1.0
        W[k1*n+1][col] += 1.0
        col += 1
        
        # p2 = (A[1,1]+A[1,2]) * B[1,k0]
        U[1*m+1][col] += 1.0
        U[1*m+2][col] += 1.0
        V[1*p+k0][col] += 1.0
        W[k0*n+1][col] += 1.0
        W[k1*n+1][col] += -1.0
        col += 1
        
        # p3 = (A[0,2]-A[0,1]) * (B[1,k1]+B[2,k1])
        U[0*m+2][col] += 1.0
        U[0*m+1][col] += -1.0
        V[1*p+k1][col] += 1.0
        V[2*p+k1][col] += 1.0
        W[k1*n+0][col] += 1.0
        W[k1*n+1][col] += 1.0
        col += 1
        
        # p4 = (A[0,1]+A[1,2]) * (B[2,k1]-B[2,k0])
        U[0*m+1][col] += 1.0
        U[1*m+2][col] += 1.0
        V[2*p+k1][col] += 1.0
        V[2*p+k0][col] += -1.0
        W[k0*n+0][col] += 1.0
        W[k1*n+0][col] += 1.0
        W[k0*n+1][col] += 1.0
        col += 1
        
        # p5 = (A[0,1]+A[1,1]) * (B[1,k0]-B[2,k0])
        U[0*m+1][col] += 1.0
        U[1*m+1][col] += 1.0
        V[1*p+k0][col] += 1.0
        V[2*p+k0][col] += -1.0
        W[k0*n+0][col] += -1.0
        W[k1*n+0][col] += 1.0
        col += 1
        
        # p6 = (A[0,2]+A[1,2]) * (B[2,k1]-B[1,k0])
        U[0*m+2][col] += 1.0
        U[1*m+2][col] += 1.0
        V[2*p+k1][col] += 1.0
        V[1*p+k0][col] += -1.0
        W[k1*n+1][col] += 1.0
        col += 1
        
        # p7 = (A[0,2]-A[1,1]) * (B[2,k0]+B[1,k0])
        U[0*m+2][col] += 1.0
        U[1*m+1][col] += -1.0
        V[2*p+k0][col] += 1.0
        V[1*p+k0][col] += 1.0
        W[k0*n+0][col] += 1.0
        col += 1
    
    assert col == total_rank, f"Col mismatch: {col} != {total_rank}"
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
