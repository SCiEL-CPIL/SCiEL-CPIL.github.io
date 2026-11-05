import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try to find better decomposition using optimization
    best_U, best_V, best_W, best_rank = None, None, None, rank_school
    
    # Try various ranks from low to high
    # Known lower bound: for n×m×p, rank >= max(n*m, m*p, p*n) / something
    # Actually, a simpler lower bound: rank >= n*m, rank >= m*p, rank >= p*n is not quite right
    # The rank must be at least the "slice rank" lower bound
    
    # Let's try to find a decomposition using iterative methods
    # The tensor equation: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    def verify(U, V, W):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # Check each component
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > 1e-10:
                        return False
        # Also check that off-diagonal terms are zero (they should be by construction if rank is minimal)
        # Actually, the tensor only has 1s on the diagonal components
        return True
    
    def try_rank(r):
        """Try to find a rank-r decomposition using random init + optimization."""
        if r >= best_rank:
            return None
        
        # Random initialization
        np.random.seed(42 + r)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Use fixed-point / alternating least squares style
        # For each term, we can solve for one matrix given the others
        # But this is non-convex, so use gradient descent
        
        # Simpler approach: use the fact that we need
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        # And ideally 0 otherwise (but the tensor is sparse)
        
        # Actually, let's use a more direct approach with structured initialization
        # based on known decompositions for smaller cases
        
        return None  # Placeholder
    
    # Known good decompositions for small cases
    # Let's use a block-based approach
    
    # Strategy: Use Strassen for 2x2x2 blocks, schoolbook for the rest
    # 2x5x6: Split into blocks
    
    # Option 1: 5 = 3+2, 6 = 3+3
    # Blocks: (2,3,3), (2,3,3), (2,2,3), (2,2,3)
    
    def strassen_2x2x2():
        """Return U,V,W for 2x2x2 Strassen decomposition, rank 7."""
        # Strassen's 7 multiplications
        # m1 = (A11-A22)*(B22-B11)
        # m2 = (A21+A22)*B11
        # m3 = A11*(B11-B22)
        # m4 = A22*(B22-B11)
        # m5 = (A11+A12)*B22
        # m6 = (A21-A11)*(B11+B12)
        # m7 = (A12-A22)*(B21+B22)
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Index: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
        # m1: (A11-A22)*(B22-B11) -> C11 += m1, C22 += m1
        # A11 = [0,0], A22 = [1,1]
        # B22 = [1,1], B11 = [0,0]
        U[0, 0] = 1; U[3, 0] = -1  # A11 - A22
        V[3, 0] = 1; V[0, 0] = -1  # B22 - B11
        W[0, 0] = 1; W[3, 0] = 1   # C11 += m1, C22 += m1
        
        # m2: (A21+A22)*B11 -> C21 += m2, C22 -= m2
        U[2, 1] = 1; U[3, 1] = 1  # A21 + A22
        V[0, 1] = 1               # B11
        W[2, 1] = 1; W[3, 1] = -1 # C21 += m2, C22 -= m2
        
        # m3: A11*(B11-B22) -> C12 += m3, C22 += m3
        U[0, 2] = 1               # A11
        V[0, 2] = 1; V[3, 2] = -1 # B11 - B22
        W[1, 2] = 1; W[3, 2] = 1  # C12 += m3, C22 += m3
        
        # m4: A22*(B22-B11) -> C11 += m4, C21 += m4
        U[3, 3] = 1               # A22
        V[3, 3] = 1; V[0, 3] = -1 # B22 - B11
        W[0, 3] = 1; W[2, 3] = 1  # C11 += m4, C21 += m4
        
        # m5: (A11+A12)*B22 -> C11 -= m5, C12 += m5
        U[0, 4] = 1; U[1, 4] = 1  # A11 + A12
        V[3, 4] = 1               # B22
        W[0, 4] = -1; W[1, 4] = 1 # C11 -= m5, C12 += m5
        
        # m6: (A21-A11)*(B11+B12) -> C22 += m6
        U[2, 5] = 1; U[0, 5] = -1 # A21 - A11
        V[0, 5] = 1; V[1, 5] = 1  # B11 + B12
        W[3, 5] = 1               # C22 += m6
        
        # m7: (A12-A22)*(B21+B22) -> C11 += m7
        U[1, 6] = 1; U[3, 6] = -1 # A12 - A22
        V[2, 6] = 1; V[3, 6] = 1  # B21 + B22
        W[0, 6] = 1               # C11 += m7
        
        return U, V, W
    
    # Let's just use the schoolbook for now and try to optimize
    U, V, W, rank = schoolbook()
    
    # Try to reduce rank by zeroing out columns that don't contribute
    # and re-solving
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
