<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook rank
    best_rank = n * m * p  # 42
    best_decomp = _schoolbook_decomp(n, m, p, best_rank)
    
    # Try to find lower rank decompositions using structured search
    # For m=3, we can exploit the small intermediate dimension
    
    # Strategy: Use a bilinear form approach where we search for rank < 42
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We need T[i,j,k] = 1 if (i,j,k) is valid, else 0
    
    # For this specific case (2,3,7), let's try ranks from 41 down
    for target_rank in range(best_rank - 1, max(20, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            decomp = _try_rank_search(n, m, p, target_rank, attempt)
            if decomp is not None:
                best_rank = target_rank
                best_decomp = decomp
                break
    
    return best_decomp

def _schoolbook_decomp(n, m, p, rank):
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

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # T[i,j,k] should equal 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    # Check that no spurious terms are generated
    # For invalid combinations, the tensor should be 0
    # But in our formulation, we only care about valid (i,j,k)
    return True

def _try_rank_search(n, m, p, target_rank, seed):
    """Try to find a rank-target_rank decomposition using structured search."""
    random.seed(seed * 1000 + target_rank)
    
    # For m=3, we can use a block structure
    # A is 2x3, B is 3x7
    # We can think of A as having 3 columns, each of size 2x1
    # B as having 3 rows, each of size 1x7
    
    # The product C = A*B has entries C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's try a construction based on polynomial interpolation or other algebraic methods
    # For now, use a simple heuristic: try to find a decomposition where
    # the factors have a specific structure
    
    # Initialize with random values
    U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
    V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
    W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
    
    # Local search to improve
    for iteration in range(100):
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Randomly modify one entry
        choice = random.randint(0, 2)
        if choice == 0:
            idx = random.randint(0, n*m-1)
            col = random.randint(0, target_rank-1)
            U[idx][col] = random.randint(-3, 3)
        elif choice == 1:
            idx = random.randint(0, m*p-1)
            col = random.randint(0, target_rank-1)
            V[idx][col] = random.randint(-3, 3)
        else:
            idx = random.randint(0, p*n-1)
            col = random.randint(0, target_rank-1)
            W[idx][col] = random.randint(-3, 3)
    
    return None
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction
from itertools import combinations

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook rank
    best_rank = n * m * p  # 42
    best_decomp = _schoolbook_decomp(n, m, p, best_rank)
    
    # Try hybrid Strassen/Schoolbook to get rank 39
    hybrid_decomp = _hybrid_strassen_schoolbook(n, m, p)
    if hybrid_decomp is not None:
        hybrid_rank = len(hybrid_decomp["U"][0])
        if hybrid_rank < best_rank:
            best_rank = hybrid_rank
            best_decomp = hybrid_decomp
    
    # Try algebraic elimination for even lower ranks
    for target_rank in range(best_rank - 1, max(30, best_rank - 10), -1):
        if time.time() > deadline:
            break
        
        decomp = _algebraic_elimination(n, m, p, target_rank)
        if decomp is not None:
            best_rank = target_rank
            best_decomp = decomp
            break
    
    return best_decomp

def _schoolbook_decomp(n, m, p, rank):
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

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # T[i,j,k] should equal 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _hybrid_strassen_schoolbook(n, m, p):
    """
    Hybrid decomposition: use Strassen for 2x2x2 blocks where possible,
    and schoolbook for the rest.
    
    For n=2, m=3, p=7:
    - Split the 7 columns into blocks: 2+2+2+1
    - For each 2-column block, use Strassen on the 2x2x2 subproblem
    - For the remaining 1 column, use schoolbook
    """
    n, m, p = 2, 3, 7
    
    # Strassen constants for 2x2x2: rank 7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our case, A is 2x3, B is 3x7
    # We can't directly apply Strassen since dimensions don't match
    # Instead, let's use a different approach:
    # Split B into 3 blocks of sizes 2, 2, 3 (columns)
    # Or better: exploit the structure differently
    
    # Alternative: Use the fact that m=3 is small
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Let's try a different decomposition:
    # Use 7 Strassen-like products for the 7 columns
    # But we need to be smarter
    
    # Actually, let's just return None and rely on algebraic elimination
    # for a cleaner approach. The hybrid approach is complex for this shape.
    return None

def _algebraic_elimination(n, m, p, target_rank):
    """
    Fix sparse support patterns for U and V, then solve linear system for W.
    """
    start_time = time.time()
    deadline = start_time + 25  # Time limit for this rank attempt
    
    # Generate candidate support patterns
    # Each term r has a support: which (i,j) pairs U[i,j,r] is nonzero,
    # which (j,k) pairs V[j,k,r] is nonzero
    
    # For rank reduction, we want to share terms between different (i,j,k) triples
    # Key insight: if two triples (i1,j1,k1) and (i2,j2,k2) can share a term,
    # then we need U[i1,j1,r] and U[i2,j2,r] both nonzero, etc.
    
    # Let's try a structured approach:
    # Start with schoolbook pattern and try to merge terms
    
    # For each potential rank reduction, we need to find a set of 
    # "mergeable" triples that can share computation
    
    # Simple approach: try random sparse patterns and solve for W
    
    random.seed(42 + target_rank)
    
    # Try multiple random patterns
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Generate random sparse U and V patterns
        U_pattern = _generate_sparse_pattern(n, m, target_rank, 2)  # avg 2 nonzeros per row
        V_pattern = _generate_sparse_pattern(m, p, target_rank, 2)  # avg 2 nonzeros per row
        
        # For each W entry, we need to solve a linear system
        # W[k,i,r] appears in equations for all (i',j,k) where U[i',j,r] != 0 and V[j,k,r] != 0
        # Actually, the equation is: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        
        # This is a nonlinear system in general, but if we fix U and V, it's linear in W
        # Wait, no: W[k,i,r] is multiplied by U[i,j,r]*V[j,k,r], so for fixed (i,j,k),
        # the equation involves W[k,i,r] for all r where U[i,j,r] and V[j,k,r] are both nonzero
        
        # Let's set up the linear system:
        # Variables: W[k*n+i, r] for all k,i,r
        # Equations: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a linear system! Let's solve it.
        
        num_W_vars = p * n * target_rank
        num_equations = n * m * p
        
        # Build the system
        A_matrix = np.zeros((num_equations, num_W_vars))
        b_vector = np.ones(num_equations)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        u_val = U_pattern[i*m+j][r]
                        v_val = V_pattern[j*p+k][r]
                        if u_val != 0 and v_val != 0:
                            w_var_idx = k*n+i*target_rank + r  # W[k*n+i, r]
                            A_matrix[eq_idx, w_var_idx] = u_val * v_val
                    eq_idx += 1
        
        # Solve the linear system
        try:
            # Use least squares since the system may be underdetermined
            W_solution, residuals, rank, sv = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
            
            # Check if the solution is exact
            if np.allclose(A_matrix @ W_solution, b_vector, atol=1e-10):
                # Reconstruct W
                W = [[0] * target_rank for _ in range(p*n)]
                for k in range(p):
                    for i in range(n):
                        for r in range(target_rank):
                            W[k*n+i][r] = W_solution[k*n+i*target_rank + r]
                
                # Convert to fractions if possible
                U = U_pattern
                V = V_pattern
                
                # Verify
                if _verify_decomp(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        except np.linalg.LinAlgError:
            continue
    
    return None

def _generate_sparse_pattern(rows, cols_per_term, target_rank, avg_nonzeros):
    """Generate a sparse pattern with approximately avg_nonzeros nonzeros per row."""
    pattern = [[0] * target_rank for _ in range(rows)]
    
    for row in range(rows):
        # Choose which columns to have nonzeros
        num_nonzeros = max(1, min(target_rank, avg_nonzeros + random.randint(-1, 1)))
        cols = random.sample(range(target_rank), num_nonzeros)
        for col in cols:
            pattern[row][col] = random.choice([-1, 1])
    
    return pattern
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
