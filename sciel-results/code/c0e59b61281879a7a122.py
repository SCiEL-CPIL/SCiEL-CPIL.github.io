"""Search-based approach for matrix multiplication tensor decomposition 4x4 by 4x7."""
import time
import random
import sys
from fractions import Fraction

DIMENSIONS = [4, 4, 7]
DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 112
    
    # Try to find a decomposition with lower rank
    # The tensor is M_{n,m,p} where (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    
    # Actually the tensor T[i,j,k] = 1 if this is a valid multiplication term
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, no. Let me re-read the contract.
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # The decomposition is: for the matrix multiplication tensor
    # T_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    
    # Actually, looking at the schoolbook:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    # So T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For schoolbook, each (i,j,k) has exactly one c where all three are 1.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for all other combinations (which shouldn't exist in this tensor), it should be 0.
    # But wait, the tensor only has entries for valid (i,j,k) where i<n, j<m, k<p.
    # The U, V, W matrices have fixed sizes, so the constraint is only for valid indices.
    
    # Let me verify: the tensor for multiplying (n x m) by (m x p) to get (n x p)
    # has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait no, that's not right either. Let me think more carefully.
    
    # The matrix multiplication tensor M_{n,m,p} is defined such that
    # (A*B)[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T has indices (i,j,k) and T[i,j,k] = 1 for all valid i,j,k.
    # The rank is the minimum r such that T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # So the constraint is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p equations in n*m*r + m*p*r + p*n*r unknowns.
    
    # Let's try a randomized approach with small integers/half-integers.
    
    def verify(U, V, W, r):
        """Check if the decomposition is valid."""
        # U: n*m x r, V: m*p x r, W: p*n x r
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def verify_fast(U, V, W, r):
        """Faster verification using the structure."""
        # Precompute for efficiency
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j]
                for k in range(p):
                    v_row = V[j*p+k]
                    w_row = W[k*n+i]
                    s = 0
                    for c in range(r):
                        s += u_row[c] * v_row[c] * w_row[c]
                    if s != 1:
                        return False
        return True
    
    # Try to find a low-rank decomposition using random search
    # Start with small ranks and increase
    
    best_result = None
    
    # First, let's try some known small ranks
    # For 4x4 * 4x7, the schoolbook is 112.
    # Let's try to find something better.
    
    # Strategy: random initialization with small values, then local search
    # or use a structured approach.
    
    # Let's try a greedy approach: build up the decomposition column by column.
    
    def try_rank(r, max_iters=50000, seed=None):
        """Try to find a valid decomposition of rank r."""
        if seed is not None:
            random.seed(seed)
        
        # Initialize with random small values
        # Use values from {-2, -1, -1/2, 0, 1/2, 1, 2} or similar
        
        def random_value():
            choices = [-2, -1, Fraction(-1,2), 0, Fraction(1,2), 1, 2]
            return random.choice(choices)
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Random initialization
            U = [[random_value() for _ in range(r)] for _ in range(n*m)]
            V = [[random_value() for _ in range(r)] for _ in range(m*p)]
            W = [[random_value() for _ in range(r)] for _ in range(p*n)]
            
            # Check if valid
            if verify_fast(U, V, W, r):
                return U, V, W, r
        
        return None
    
    # Also try a more structured approach
    def try_structured(r, max_iters=100000, seed=None):
        """Try structured initialization."""
        if seed is not None:
            random.seed(seed)
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Initialize with a pattern
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            
            # Assign random non-zero values to some positions
            for c in range(r):
                # Pick random positions
                ui = random.randint(0, n*m-1)
                vj = random.randint(0, m*p-1)
                wk = random.randint(0, p*n-1)
                
                # Make sure the indices are consistent (same j for U and V)
                # U index: i*m + j, V index: j*p + k, W index: k*n + i
                # For consistency, we need the same j in U and V, same k in V and W, same i in U and W
                # This is tricky for random assignment.
                
                # Instead, let's just use random values
                U[ui][c] = random.choice([-1, 1, Fraction(1,2), -Fraction(1,2)])
                V[vj][c] = random.choice([-1, 1, Fraction(1,2), -Fraction(1,2)])
                W[wk][c] = random.choice([-1, 1, Fraction(1,2), -Fraction(1,2)])
            
            if verify_fast(U, V, W, r):
                return U, V, W, r
        
        return None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But that's a very weak bound. Let's try from 28 upward.
    
    # Actually, let's first check if the schoolbook works (it should)
    schoolbook_U = [[0]*schoolbook_rank for _ in range(n*m)]
    schoolbook_V = [[0]*schoolbook_rank for _ in range(m*p)]
    schoolbook_W = [[0]*schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_U[i*m+j][col] = 1
                schoolbook_V[j*p+k][col] = 1
                schoolbook_W[k*n+i][col] = 1
                col += 1
    
    if verify_fast(schoolbook_U, schoolbook_V, schoolbook_W, schoolbook_rank):
        best_result = (schoolbook_U, schoolbook_V, schoolbook_W, schoolbook_rank)
    
    # Now try to find better
    # Try ranks from 28 to schoolbook_rank-1
    for r in range(28, min(schoolbook_rank, 112)):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try random search
        result = try_rank(r, max_iters=10000, seed=r*1000)
        if result is not None:
            best_result = result
            break
        
        # Try structured
        result = try_structured(r, max_iters=50000, seed=r*2000)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W, r = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": schoolbook_U, "V": schoolbook_V, "W": schoolbook_W}

