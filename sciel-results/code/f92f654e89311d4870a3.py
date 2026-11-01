"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # The tensor for matrix multiplication C = A @ B where A is n x m, B is m x p
    # has indices T[i, j, k] = sum_l delta_{i+l, j} ... 
    # Actually, the standard decomposition is:
    # (A @ B)_{i,k} = sum_{j=0}^{m-1} A_{i,j} * B_{j,k}
    # The rank-r decomposition gives:
    # (A @ B)_{i,k} = sum_{r=0}^{rank-1} W_{k, r, i} * prod_{j=0}^{m-1} (U_{i, j, r} * A_{i,j}) * (V_{j, k, r} * B_{j,k})
    # Wait, the standard form is:
    # M_{i,k}(A,B) = sum_r W_{i,k,r} * prod_j U_{i,j,r}(A_{i,j}) * V_{j,k,r}(B_{j,k})
    # But U is (n*m, rank) so U[i, j] for A_{i,j} means U row index is i*m+j? No.
    # Looking at the seed: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # W[k*n+i, r] -> so W is indexed by output (k, i) where i is row of C, k is col of C
    # U[i, r] for A_{i, j}? No, U is (n*m, rank) so row index is i*m + j for A_{i,j}
    # V[j, k, r] for B_{j,k}? V is (m*p, rank) so row index is j*p + k for B_{j,k}
    
    # Let's verify by checking: for all i, j, k, the coefficient of A_{i,j} * B_{j,k} in the decomposition
    # should be 1 if this is the only term contributing to C_{i,k} from that (j) index, but actually
    # the decomposition computes C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The decomposition computes: sum_r W[k*n+i, r] * (prod over j? No, it's linear)
    
    # Actually the standard tensor decomposition for matrix multiplication:
    # T_{i,j,k} = delta_{i,j} (for the identity part) ... no.
    # The multiplication tensor M_{n,m,p} has entries M_{i,j,k} = 1 if the output position (i,k)
    # depends on input A_{i,?} and B_{?,k}. 
    # More precisely: C_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}
    # The tensor is: M[i, j, k] where i indexes A row, j indexes the shared dimension, k indexes B col
    # M[i, j, k] = 1 if this is a valid term, i.e., always 1 for the term A_{i,j} B_{j,k}
    # The rank-r decomposition: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right either. Let me re-examine.
    
    # In the seed code:
    # U[0][0] = 1 means U row 0, col 0 = 1. Row 0 corresponds to A[0,0] (i=0, j=0)
    # V[0][0] = 1 means V row 0, col 0 = 1. Row 0 corresponds to B[0,0] (j=0, k=0)
    # W[0][0] = 1 means W row 0, col 0 = 1. Row 0 corresponds to C[0,0] (i=0, k=0)
    
    # So the decomposition is: C[i,k] = sum_r W[k*n+i, r] * (sum over terms...)
    # Actually no. The bilinear form is:
    # C[i,k] = sum_r (sum_{j'} U[i*m + j', r] * A[i,j']) * (sum_{j''} V[j''*p + k, r] * B[j'',k]) * W[k*n+i, r]
    # No wait, that would be a product of sums which is quadratic.
    
    # Let me look at this more carefully. The standard format is:
    # C[i,k] = sum_r W_{i,k,r} * prod_{j=0}^{m-1} [U_{i,j,r} * A_{i,j}] * [V_{j,k,r} * B_{j,k}]
    # But U is 2D (n*m, rank), not 3D. So U[i, r] is a scalar.
    
    # I think the correct interpretation is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_{j=0}^{m-1} U[i*m+j, r] * A[i,j]) * (sum_{j=0}^{m-1} V[j*p+k, r] * B[j,k])
    # This is a bilinear form where for each rank term r, we have:
    # - A linear form in A: L_r^A(A) = sum_j U[i*m+j, r] * A[i,j]  (note: depends on i as well)
    # - A linear form in B: L_r^B(B) = sum_j V[j*p+k, r] * B[j,k]  (note: depends on k as well)
    # - And W[k*n+i, r] is a scalar weight
    
    # Wait, but U[i*m+j, r] depends on both i and j, so for a fixed i, we sum over j.
    # Similarly V[j*p+k, r] depends on j and k, so for fixed k, we sum over j.
    
    # Let me verify with the seed:
    # M1: U[0][0]=1 (A[0,0]), U[3][0]=1 (A[1,1]) -> L = A[0,0] + A[1,1]
    # V[0][0]=1 (B[0,0]), V[4][0]=1 (B[1,1]) -> L = B[0,0] + B[1,1]
    # W[0][0]=1 -> C[0,0] gets contribution (A[0,0]+A[1,1])(B[0,0]+B[1,1]) * 1
    # C[0,0] should be A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # Hmm, (A[0,0]+A[1,1])(B[0,0]+B[1,1]) = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
    # That doesn't directly give C[0,0]. But combined with other terms it should work out.
    
    # The verification should check that the bilinear form exactly reproduces matrix multiplication.
    # For a given (i, k), the computed value is:
    # sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # This should equal sum_j A[i,j] * B[j,k]
    
    # We can verify this by picking random A and B and checking.
    rng = np.random.default_rng(42)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(len(W[0])):
                lA = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                lB = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                C_computed[i, k] += W[k*n+i, r] * lA * lB
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen-style part for the 2x2 block (columns 0 and 1 of B)
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # W for Schoolbook outputs
    # C02 = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    # C12 = M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    # Verify the decomposition
    if not _verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 12
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
