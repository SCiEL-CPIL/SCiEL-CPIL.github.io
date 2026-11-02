"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T has dimensions (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, else 0
    
    # We want to find a rank-r decomposition:
    # T = sum_{c=1}^{r} U[:,c] * V[:,c] * W[:,c]
    # where U is (n*m x r), V is (m*p x r), W is (p*n x r)
    
    # For 3x3 matrix multiplication, the known minimal rank is 23 (Landsberg)
    # but finding explicit decompositions is hard. Let's try to find one.
    
    # Strategy: Use a randomized approach with verification
    # We'll try ranks from 23 upward and use a combination of:
    # 1. Random initialization with rational entries
    # 2. Gradient-free optimization (since we need exact rational solutions)
    
    # Actually, let's use a more systematic approach.
    # The tensor can be viewed as a 3-way array. We can use the fact that
    # for small dimensions, we can try to find low-rank decompositions.
    
    # Let's first verify the schoolbook rank-27 solution works, then try to improve.
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a rank-r decomposition using an iterative approach
    # We'll use the fact that if we fix two factors, the third is determined
    # by linear equations.
    
    best_rank = n * m * p  # Start with schoolbook
    
    # First, let's implement a verification function
    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        r = U.shape[1]
        # Compute the decomposition
        # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    val = sum(U[i,c] * V[j,c] * W[k,c] for c in range(r))
                    if abs(val - T[i,j,k]) > 1e-10:
                        return False
        return True
    
    def try_rank(r, max_iters=200):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 10:
            return None
        
        # Random initialization with small rational values
        # Use values from {-1, 0, 1} and their halves
        vals = [-1, -0.5, 0, 0.5, 1]
        
        for attempt in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            U = np.array([[Fraction(Fraction(vals[np.random.randint(len(vals))])) for _ in range(r)] for _ in range(n*m)], dtype=object)
            V = np.array([[Fraction(Fraction(vals[np.random.randint(len(vals))])) for _ in range(r)] for _ in range(m*p)], dtype=object)
            
            # For each (i,j,k) where T[i,j,k] != 0, we need:
            # sum_c U[i,c]*V[j,c]*W[k,c] = 1
            # For each (i,j,k) where T[i,j,k] == 0, we need:
            # sum_c U[i,c]*V[j,c]*W[k,c] = 0
            
            # This is a system of equations in W. Let's set up the system.
            # For fixed (i,j), the equation is:
            # sum_c (U[i,c]*V[j,c]) * W[k,c] = T[i,j,k]
            
            # Collect all constraints
            # Variables: W[k,c] for k in range(p*n), c in range(r)
            # Total variables: p*n*r
            
            # Let's build the linear system
            # For each (i,j,k): sum_c U[i,c]*V[j,c]*W[k,c] = T[i,j,k]
            
            num_eqs = 0
            eqs = []
            rhs = []
            
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        # Coefficients for W[k,c]
                        coeffs = []
                        for c in range(r):
                            coeffs.append(U[i,c] * V[j,c])
                        eqs.append(coeffs)
                        rhs.append(Fraction(T[i,j,k]))
                        num_eqs += 1
            
            # Solve the linear system using Gaussian elimination with Fractions
            # This is expensive, so let's be careful
            
            # Actually, this approach has p*n*r = 9*r unknowns and 27*9 = 243 equations
            # For r=23, that's 207 unknowns and 243 equations - overdetermined
            
            # Let's use a different approach: iterate on W given U,V, then on U,V given W
            # This is like alternating minimization but for exact solutions
            
            # Alternative: use the fact that we can fix some structure
            
            # Let's try a simpler approach: use known results
            # For 3x3, there's a known rank-23 decomposition by Landsberg
            # but it's complex. Let's try to find it via search.
            
            # Actually, let's use a more practical approach:
            # Start from the rank-27 schoolbook and try to merge terms
            
            break
        
        return None
    
    # Let's use a different strategy: start from schoolbook and try to reduce rank
    # by finding linear dependencies
    
    # The schoolbook decomposition has 27 terms. Each term corresponds to (i,j,k).
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for term c corresponding to (i,j,k)
    
    # To reduce rank, we need to find a linear combination of terms that gives zero
    # or find that some terms can be combined.
    
    # Actually, let's just return the schoolbook solution for now, since finding
    # a lower rank decomposition is non-trivial.
    
    # But wait - the directive says to find rank-minimal. Let's try harder.
    
    # Known result: The rank of 3x3 matrix multiplication is 23 (Landsberg, 1997)
    # but explicit decompositions are hard to find.
    
    # Let's try a randomized approach with more structure.
    
    # Actually, let's try using numpy to solve the linear system for W given U and V.
    
    def find_decomposition_rank(r, num_trials=50):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 10:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            # Initialize U and V randomly with small integers
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            
            for i in range(n*m):
                for c in range(r):
                    U[i,c] = np.random.randint(-2, 3)
            for j in range(m*p):
                for c in range(r):
                    V[j,c] = np.random.randint(-2, 3)
            
            # Build the linear system for W
            # For each (i,j,k): sum_c U[i,c]*V[j,c]*W[k,c] = T[i,j,k]
            
            # The unknowns are W[k,c] for k in 0..8, c in 0..r-1
            # Total: 9*r unknowns
            
            # Equations: 27*9 = 243 equations
            
            A = np.zeros((243, 9*r))
            b = np.zeros(243)
            
            eq_idx = 0
            for i in range(9):
                for j in range(9):
                    for k in range(9):
                        for c in range(r):
                            A[eq_idx, k*r + c] = U[i,c] * V[j,c]
                        b[eq_idx] = T[i,j,k]
                        eq_idx += 1
            
            # Solve the least squares problem (overdetermined)
            # Check if there's an exact solution
            
            try:
                W, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is exact
                if len(residuals) > 0 and residuals[0] < 1e-10:
                    # Verify
                    if verify_decomposition(U, V, W):
                        return U, V, W
            except:
                pass
        
        return None
    
    # Try ranks from 23 to 27
    for r in range(23, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = find_decomposition_rank(r, num_trials=30)
        if result is not None:
            U, V, W = result
            best_rank = r
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Round to nearest integer or half-integer if close
            def round_entries(mat):
                mat_r = []
                for row in mat:
                    row_r = []
                    for val in row:
                        # Round to nearest 0.5
                        rounded = round(val * 2) / 2
                        row_r.append(rounded)
                    mat_r.append(row_r)
                return mat_r
            
            U_list = round_entries(U_list)
            V_list = round_entries(V_list)
            W_list = round_entries(W_list)
            
            # Final verification
            if verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list)):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
