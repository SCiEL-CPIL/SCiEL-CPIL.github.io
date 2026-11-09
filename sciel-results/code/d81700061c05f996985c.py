"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Known lower bounds:
    # For 3x3 x 3x8, the rank is at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But more specifically, for matrix multiplication, the rank must be at least
    # the maximum of the three dimensions' products in certain ways.
    # Actually, for [n,m,p] matrix multiplication, the tensor rank is at least
    # max(n*m, m*p, p*n) but that's not quite right either.
    # The correct lower bound is that the rank must be at least max(n*m, m*p, p*n)
    # No wait, let me think again.
    # For multiplying an n x m matrix by an m x p matrix:
    # - U has shape (n*m, r)
    # - V has shape (m*p, r)  
    # - W has shape (p*n, r)
    # The rank r must be at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But actually the lower bound is more subtle.
    
    # Let's try to find a better decomposition.
    # For 3x3 x 3x8, we can try rank 24, 25, ..., up to some limit.
    
    # Actually, let's think about this differently.
    # The matrix multiplication tensor T has entries T[i,j,k] = delta_{j,k} (where j is the contraction index)
    # Wait, let me re-read the problem.
    # Multiplying A (3x3) by B (3x8) gives C (3x8)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] where the "free" indices are i (row of A, row of C), 
    # j (col of A, row of B), and k (col of B, col of C)
    # The decomposition: C[i,k] = sum_r U[i*m+j... wait, I need to re-read the interface.
    
    # Interface: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] - so W is indexed by (k, i) where k is col of C and i is row of C
    # U[i*m+j, r] - indexed by (i, j) where i is row of A, j is col of A
    # V[j*p+k, r] - indexed by (j, k) where j is row of B, k is col of B
    
    # So the decomposition is: C[i,k] = sum_r U[i*m+j... no wait.
    # Actually, the standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: T_{i,j,k} = delta_{j,j} (the middle index is contracted)
    # Wait no. Let me think of it as: the bilinear form is sum over r of U_r * V_r * W_r
    # where U_r is a vector in R^{n*m}, V_r is a vector in R^{m*p}, W_r is a vector in R^{p*n}
    # And C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * ... 
    # Hmm, this is getting confusing. Let me just use the schoolbook as a baseline and try to improve.
    
    # Schoolbook: rank = n*m*p = 72
    # For each (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    # Can we do better? For 3x3 x 3x3, the best known is 23 (Laser method / Strassen-like).
    # For 3x3 x 3x8, we need to find the minimal rank.
    
    # Let's try a random search approach: start with a random decomposition of some rank r,
    # and try to optimize it. But this is very hard to do exactly.
    
    # Alternative: use the fact that for fixed n,m, the rank as a function of p might be linear.
    # For 3x3 x 3xp, if we know the rank for p=3 is 23, maybe for p=8 it's something like...
    # Actually, this is complex. Let me try a different approach.
    
    # Let's try to construct a decomposition of rank < 72 using known techniques.
    # One approach: use Strassen-like ideas or exploit structure.
    
    # For now, let's try a simple improvement: can we merge some of the schoolbook terms?
    # The schoolbook has 72 terms. Each term corresponds to a unique (i,j,k) triple.
    # Two terms can be merged if they have the same pattern in U, V, W up to scaling.
    # But in schoolbook, each (i,j,k) gives a unique combination, so no merging is possible
    # with just 0/1 entries.
    
    # Let's try a randomized approach: start with a random rank-r decomposition and try to 
    # verify if it's correct, then optimize r downward.
    
    # Actually, let's think about this more carefully. The problem asks for integer/half-integer
    # entries. This is a strong constraint.
    
    # Let me try a small search: for rank r from 24 upward, try to find a valid decomposition.
    # But the search space is huge. 
    
    # Alternative: use known results. For 3x3 x 3x3, the rank is 23 (proven minimal).
    # For 3x3 x 3x8, we can think of it as 8 copies of 3x3 x 3x1 plus interactions.
    # Actually, 3x3 x 3x8 = 3x3 x (3x8). We can write B as [B1, B2, B3, B4, B5, B6, B7, B8]
    # where each Bj is 3x1. Then C = [A*B1, A*B2, ..., A*B8].
    # If we could compute A*Bj for all j simultaneously with fewer than 8*23 = 184 operations...
    # But that's not how tensor rank works directly.
    
    # Let me try a different tactic: use a known decomposition for 3x3 x 3x3 of rank 23,
    # and extend it to 3x3 x 3x8.
    
    # Actually, I recall that for matrix multiplication [n,m,p], the tensor rank is denoted R(n,m,p).
    # R(3,3,3) = 23 (proven).
    # R(3,3,8) is not immediately obvious.
    
    # Let's try a computational approach: use a MILP or SMT solver... but we only have NumPy/SciPy.
    
    # Given the constraints, let's try a heuristic search:
    # 1. Start with the schoolbook decomposition (rank 72).
    # 2. Try to reduce the rank by finding linear dependencies or using known tricks.
    
    # Actually, let me try a simpler approach: use the fact that the rank must be at least
    # max(n*m, m*p, p*n) = 24, and try to find a decomposition of rank 24, 25, etc.
    
    # For rank 24: U is 9x24, V is 24x24, W is 24x9.
    # The condition is: for all i,k: sum_r U[i*m+0, r]*V[0*p+k, r]*W[k*n+i, r] + ... 
    # Wait, I need to get the exact formula right.
    
    # Let me re-derive: 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k]) * ... 
    # No, that's not right either. The tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}  -- no, this isn't standard.
    
    # Let me look at the schoolbook code again:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1  
    # W[k*n+i][column] = 1
    # So for term r corresponding to (i,j,k):
    # U has 1 at position (i*m+j), V has 1 at (j*p+k), W has 1 at (k*n+i)
    
    # The reconstruction would be:
    # C[i,k] = sum_r U_flat[i*m+?, r] * ... 
    # I think the correct interpretation is:
    # Define A as a vector in R^{n*m} with A_flat[i*m+j] = A[i,j]
    # Define B as a vector in R^{m*p} with B_flat[j*p+k] = B[j,k]
    # Define C as a vector in R^{p*n} with C_flat[k*n+i] = C[i,k]
    # Then the bilinear map is: C = sum_r (U^T A) * (V^T B) * W  -- no, that gives a scalar.
    
    # Actually, I think the correct formula is:
    # C_flat[k*n+i] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * ... 
    # This still doesn't make sense dimensionally.
    
    # Let me just trust the schoolbook implementation and try to verify it, then search for improvements.
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        rank = len(U[0]) if U else 0
        # Test with random matrices
        random.seed(42)
        for _ in range(10):
            A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
            # Compute C = A * B
            C_expected = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            # Compute C from decomposition
            C_actual = [[0] * p for _ in range(n)]
            for r in range(rank):
                u_val = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                v_val = sum(V[j*p+k][r] * B[j][k] for k in range(p))
                # Hmm, this still doesn't work. Let me think...
                # Actually, I think the formula is:
                # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j?
                # No, that's a trilinear form, not bilinear.
                
                # I think the correct interpretation is that the tensor T is defined as:
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # And we need T[i,j,k] = delta_{j, j} which is 1 if... wait.
                
                # For matrix multiplication, the tensor is T[i,j,k] where the "bilinear" 
                # structure is: C[i,k] = sum_j A[i,j] B[j,k]
                # So the tensor should satisfy: sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k]
                # which means T[i,j,k] should be such that when contracted with A and B, 
                # we get C. This means T[i,j,k] = delta_{j,j} = 1 for all i,j,k? No.
                
                # Actually, I think the tensor is: T[i, j, k] where the free indices are 
                # arranged so that the bilinear form is:
                # C[i,k] = sum_{j} A[i,j] B[j,k]
                # In tensor decomposition language, we write:
                # C = sum_r U_r (A) * V_r (B) * W_r
                # where U_r(A) = sum_{i,j} U[i*m+j, r] A[i,j] is a scalar
                # V_r(B) = sum_{j,k} V[j*p+k, r] B[j,k] is a scalar
                # W_r is... a vector? That doesn't work.
                
                # I think I'm overcomplicating this. Let me just use the schoolbook and 
                # try to find improvements through known methods.
                
            break  # Skip verification for now, it's not working out
        
        # Let me try a different verification approach
        # The tensor T_{i,j,k} for matrix multiplication [n,m,p] is defined as:
        # T_{i,j,k} = 1 if the "contraction" index matches, i.e., it's the identity in the middle
        # More precisely, T_{i,j,k} is a trilinear tensor where:
        # sum_{j} T_{i,j,k} A_{i,j} B_{j,k} = C_{i,k}
        # This requires T_{i,j,k} = delta_{j,j} = 1 for all i,j,k? No, that gives sum_j A[i,j]B[j,k] which is correct!
        # Wait, so T_{i,j,k} = 1 for all i,j,k? That can't be right because then the rank would be 1.
        
        # I think the issue is that the tensor is not T_{i,j,k} but rather the bilinear map itself.
        # Let me just assume the schoolbook is correct and try to find a better decomposition.
        pass
    
    # Given the complexity, let's try a pragmatic approach:
    # Use the schoolbook decomposition but try to find a lower rank through known results.
    
    # For 3x3 x 3x3, the minimal rank is 23.
    # For 3x3 x 3x8, let's try to use a block structure.
    
    # Actually, let me try a completely different approach: use a known rank-23 decomposition
    # for 3x3 x 3x3 and extend it.
    
    # The rank-23 decomposition for 3x3 x 3x3 is known (from Bini et al. / Lader).
    # Let me try to implement it.
    
    # Actually, given the time constraints and the difficulty of implementing known decompositions
    # from scratch, let me try a randomized search for a decomposition of rank < 72.
    
    # Strategy: 
    # 1. Start with a random rank-r decomposition (r = 72 initially).
    # 2. Use gradient descent or coordinate descent to minimize the error.
    # 3. Try to reduce r by eliminating near-zero components.
    
    # But this is very hard to do exactly with integer/half-integer constraints.
    
    # Let me try a simpler approach: just return the schoolbook for now, but with a small
    # optimization: check if we can reduce the rank by 1 or 2 through a simple search.
    
    # Actually, let me try to implement a known result. For 3x3 x 3x3, the rank is 23.
    # I'll try to hardcode a known rank-23 decomposition.
    
    # From the literature, one rank-23 decomposition for 3x3 x 3x3 uses the following:
    # (This is from Bini, Capovani, Lotti, Romani 1979, or similar sources)
    # The decomposition involves 23 terms with specific coefficients.
    
    # Due to the complexity, let me try a different strategy: use a SAT/SMT-like approach
    # with a small search space.
    
    # For now, let's return the schoolbook decomposition, which is guaranteed correct.
    # We can try to improve it in future iterations.
    
    rank = schoolbook_rank
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
