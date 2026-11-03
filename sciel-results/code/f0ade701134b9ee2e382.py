"""Find rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import time
import sys
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best = {'U': U_sb, 'V': V_sb, 'W': W_sb, 'rank': rank_sb}
    
    # Try to find lower rank decompositions
    # Strategy: Use the fact that for matrix multiplication, we can potentially
    # share computation across different (i,j,k) triples.
    
    # For small matrices, we can try a direct search approach.
    # The key insight: each term t contributes U[i,j,t]*V[j,k,t]*W[k,i,t] to C[i,k].
    # We need: for all i,k: C[i,k] = sum_j A[i,j]*B[j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t]
    
    # This means: sum_t (sum_j U[i,j,t]*V[j,k,t]) * W[k,i,t] = sum_j A[i,j]*B[j,k]
    
    # Let's define M_t[i,k] = sum_j U[i,j,t]*V[j,k,t]
    # Then we need: sum_t M_t[i,k]*W[k,i,t] = (A*B)[i,k]
    
    # This is a bilinear form decomposition.
    
    # For exact search with small dimensions, let's try a constructive approach:
    # We'll attempt ranks from a lower bound up to the schoolbook rank.
    
    # Lower bound: The tensor has non-zero structure. A simple lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But better bounds exist. For now, let's try to improve from schoolbook.
    
    # Approach: Try to find a decomposition by solving for factors given a target rank.
    # We'll use a greedy approach: iteratively add rank-1 terms that best approximate the residual.
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W with given rank exactly decompose the tensor."""
        n, m, p = DIMENSIONS
        # Check: for all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] should equal 1
        # (since the tensor T[i,j,k] = 1 for valid i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if s != 1:
                        return False
        return True
    
    def verify_via_matrix_mult(U, V, W, rank):
        """Verify by checking that the decomposition reproduces matrix multiplication.
        For any A (n x m) and B (m x p), C = A*B should equal sum_t (U_t * A) * (B * W_t) where
        U_t is the t-th column reshaped, etc.
        Actually, the correct check: C[i,k] = sum_j A[i,j] B[j,k]
        And our decomposition gives: C[i,k] = sum_t (sum_j U[i,j,t] A[i,j]) ... no.
        
        Let me reconsider. The tensor decomposition T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
        means that for the bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j]B[j,k],
        we have C[i,k] = sum_t (sum_j U[i,j,t] A[i,j]) * (sum_j V[j,k,t] B[j,k]) * W[k,i,t]
        
        Wait, that's not right either. Let me re-derive.
        
        The matrix multiplication tensor T satisfies: T(A,B) = A*B, i.e., (A*B)[i,k] = sum_j A[i,j] B[j,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
        with T[i,j,k] = 1.
        
        A rank-r decomposition: T[i,j,k] = sum_{t=1}^r U[i,j,t] V[j,k,t] W[k,i,t]
        Then: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_t (sum_{i,j} U[i,j,t] A[i,j]) (sum_{j,k} V[j,k,t] B[j,k]) W[k,i,t]
        
        But this doesn't directly give C[i,k]. The issue is that W depends on both k and i.
        
        Actually, the standard formulation: the multiplication tensor as a 3-tensor T[i,j,k]
        has the property that the bilinear map is: for each output position (i,k),
        C[i,k] = sum_j A[i,j] B[j,k].
        
        A rank-1 term in the tensor is u[i,j] * v[j,k] * w[k,i].
        The contribution to C[i,k] from term t is: w[k,i,t] * (sum_j u[i,j,t] * A[i,j]) * ... no.
        
        Let me think differently. The tensor T[i,j,k] = 1 means:
        C = A * B where C[i,k] = sum_j A[i,j] B[j,k].
        
        If we have T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t], then:
        sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_t sum_{i,j,k} U[i,j,t] V[j,k,t] W[k,i,t] A[i,j] B[j,k]
        = sum_t (sum_{i,j} U[i,j,t] A[i,j]) (sum_{j,k} V[j,k,t] B[j,k]) ... but W[k,i,t] couples i and k.
        
        This is getting confused. Let me use a different approach.
        
        The correct interpretation: The tensor T[i,j,k] defines the bilinear map via:
        (A, B) -> C where C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        
        For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
        
        A rank-r decomposition of T means: T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
        Then the bilinear map becomes:
        C[i,k] = sum_j (sum_t U[i,j,t] V[j,k,t] W[k,i,t]) A[i,j] B[j,k]
        = sum_t W[k,i,t] (sum_j U[i,j,t] A[i,j]) (sum_j V[j,k,t] B[j,k])
        
        Wait, that's still not quite right because the sums over j are separate.
        
        Actually: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j sum_t U[i,j,t] V[j,k,t] W[k,i,t] A[i,j] B[j,k]
        = sum_t W[k,i,t] sum_j U[i,j,t] V[j,k,t] A[i,j] B[j,k]
        
        This is a quadratic form in A and B for each t, which doesn't factor nicely.
        
        I think I'm overcomplicating this. Let me just use the direct verification:
        The decomposition is valid if for all i,j,k: sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1
        """
        return verify_decomposition(U, V, W, rank)
    
    # Given the complexity of exact search, let's try a constructive approach
    # for specific small cases.
    
    # For 3x4x5, let's try to find a decomposition with rank < 60.
    # One approach: use the fact that we can group terms.
    
    # Actually, let's try a different strategy: use known results or a heuristic
    # that tries to minimize rank by allowing shared factors.
    
    # Heuristic: Start with schoolbook, then try to "merge" pairs of terms
    # where possible by solving for a combined term.
    
    # For now, let's try a simple improvement: if we can find that some
    # (i,j,k) triples can share a rank-1 term, we reduce the rank.
    
    # A rank-1 term U[i,j]V[j,k]W[k,i] contributes to ALL (i',j',k') where
    # U[i',j'] != 0, V[j',k'] != 0, W[k',i'] != 0.
    
    # So a sparse U,V,W can cover multiple tensor entries with one term.
    
    # Let's try: for each possible "pattern" of non-zero entries in U, V, W,
    # check if it can cover a set of (i,j,k) triples.
    
    # This is a set cover problem, which is NP-hard, but for small instances
    # we can try.
    
    # Given time constraints, let's try a few specific constructions.
    
    # Construction 1: Try to use a decomposition based on the fact that
    # 3x4 and 4x5 can be embedded in larger structures.
    
    # For now, let's just return the schoolbook if we can't find better quickly.
    # But let's try one more thing: a greedy algorithm.
    
    def greedy_decomposition(max_rank=None):
        """Try to build a low-rank decomposition greedily."""
        if max_rank is None:
            max_rank = rank_sb
        
        n, m, p = DIMENSIONS
        
        # We'll try to find U, V, W of given rank by solving a sequence of linear systems.
        # For each rank r, we check if there exist U (n*m x r), V (m*p x r), W (p*n x r)
        # such that for all (i,j,k): sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1
        
        # This is a system of n*m*p equations in 3*r*(n*m + m*p + p*n) variables (but coupled).
        
        # For small r, we can try random initialization and Newton's method,
        # but exactness is hard to guarantee.
        
        # Alternative: use the fact that we can fix some variables and solve for others.
        
        # Let's try a specific construction for rank < 60.
        
        # Idea: Use a block structure. For example, if we can decompose the
        # 4-dimensional middle index into two parts of size 2, we might get
        # a Strassen-like improvement.
        
        # For 2x2x2, Strassen gives rank 7 vs schoolbook 8.
        # For 3x3x3, best known is 23 vs schoolbook 27.
        
        # For 3x4x5, let's try to see if we can get below 60.
        
        # Simple upper bound improvement: if we can find that the tensor has
        # a decomposition of rank r, we're done.
        
        # Let's try rank from 20 up to 60 and check feasibility.
        
        # For each candidate rank r, we can try to solve the system.
        # The system is: for each (i,j,k), sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1
        
        # This is a polynomial system. For exact solutions with rational numbers,
        # we can use Groebner bases, but that's expensive.
        
        # Instead, let's try a constructive approach: assume U and V are fixed
        # (e.g., sparse), and solve for W.
        
        # Fix U and V: then for each (i,k), we have:
        # sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] V[j*p+k, t]) = 1
        # Let M_t[i,k] = sum_j U[i*m+j, t] V[j*p+k, t]
        # Then: sum_t W[k*n+i, t] M_t[i,k] = 1
        
        # This is a linear system in W for each (i,k) pair, but W[k*n+i, t] is shared
        # across different (i,k) for the same (k,i) ... wait, W is indexed by (k,i),
        # so W[k*n+i, t] is a separate variable for each (k,i,t).
        
        # Actually, for fixed (i,k), the equation is:
        # sum_t W[k*n+i, t] * M_t[i,k] = 1
        # This is a single equation in r unknowns W[k*n+i, 0], ..., W[k*n+i, r-1].
        
        # If r >= 1, we can always satisfy this by choosing W appropriately,
        # AS LONG AS M_t[i,k] is not all zero for the t where W is non-zero.
        
        # So the constraint is: for each (i,k), the vector (M_0[i,k], ..., M_{r-1}[i,k])
        # must not be the zero vector (so we can find W[k*n+i, :] to satisfy the equation).
        
        # Additionally, we need W entries to be integer or half-integer.
        
        # So the problem reduces to: find sparse U (n*m x r) and V (m*p x r) such that
        # for all (i,k): M[i,k] = (sum_j U[i*m+j, 0] V[j*p+k, 0], ..., sum_j U[i*m+j, r-1] V[j*p+k, r-1]) != 0
        # and then W can be solved to give integer/half-integer values.
        
        # For W to be integer/half-integer, we need M_t[i,k] to be such that
        # W[k*n+i, t] = 1 / (some combination) is rational with denominator 1 or 2.
        
        # This is getting complex. Let's just try a few specific constructions.
        
        return None  # Placeholder
    
    # Given the time constraints and complexity, let's try a different approach:
    # Use known results or a simple heuristic.
    
    # For 3x4x5 matrix multiplication, the schoolbook rank is 60.
    # It's likely that a lower rank is possible, but finding it exactly is hard.
    
    # Let's try a simple improvement: if we can find that the middle dimension (4)
    # can be split, we might get a Strassen-like gain.
    
    # For now, let's return the schoolbook decomposition as a safe baseline.
    # The search for better decompositions would require more sophisticated methods.
    
    # Check deadline
    if time.time() > deadline:
        return {'U': best['U'], 'V': best['V'], 'W': best['W']}
    
    # Try a few specific low-rank constructions
    # Construction: try rank 20 (the lower bound)
    # If we can find U, V, W of rank 20 that work, great.
    
    # For rank r, we need n*m*p = 60 equations in 3*r*(n*m + m*p + p*n) = 3*r*47 variables.
    # For r=20: 3*20*47 = 2820 variables, 60 equations. Should be feasible if the system is consistent.
    
    # Let's try to solve for rank 20 by fixing U and V to be simple (e.g., identity-like)
    # and solving for W.
    
    def try_rank(r):
        """Try to find a decomposition of rank r."""
        n, m, p = DIMENSIONS
        
        # Fix U to be a sparse matrix: U[i*m+j, t] = 1 if t == i*m+j else 0
        # This means U is an (n*m) x r matrix with 1s on the "diagonal" if r >= n*m
        # For r < n*m, we need a different strategy.
        
        # For r >= n*m = 12, let's set U[i*m+j, t] = 1 if t == i*m+j else 0
        # Then M_t[i,k] = sum_j U[i*m+j, t] V[j*p+k, t] = V[t*p + (j where i*m+j == t), k] if t < n*m
        # This is messy.
        
        # Alternative: use a random sparse U and V and check if W can be solved.
        
        import random
        random.seed(42)
        
        for trial in range(100):
            # Generate random sparse U, V with entries in {-1, 0, 1}
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            
            for i in range(n * m):
                for t in range(r):
                    if random.random() < 0.3:
                        U[i][t] = random.choice([-1, 1])
            
            for j in range(m * p):
                for t in range(r):
                    if random.random() < 0.3:
                        V[j][t] = random.choice([-1, 1])
            
            # Now solve for W: for each (i,k), sum_t W[k*n+i, t] * M_t[i,k] = 1
            # where M_t[i,k] = sum_j U[i*m+j, t] * V[j*p+k, t]
            
            W = [[0] * r for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Compute M_t[i,k] for t = 0..r-1
                    M = [0] * r
                    for t in range(r):
                        for j in range(m):
                            M[t] += U[i * m + j][t] * V[j * p + k][t]
                    
                    # Solve: sum_t W[k*n+i, t] * M[t] = 1
                    # This is a single equation in r unknowns.
                    # If r >= 1, we can set W[k*n+i, 0] = 1/M[0] if M[0] != 0,
                    # and W[k*n+i, t] = 0 for t > 0.
                    # But we need W to be integer or half-integer.
                    
                    # Check if M has a non-zero entry
                    non_zero = [t for t in range(r) if M[t] != 0]
                    if not non_zero:
                        valid = False
                        break
                    
                    # Try to find W[k*n+i, :] such that sum_t W[t]*M[t] = 1
                    # and W[t] is in {0, 1/2, 1, -1/2, -1, ...}
                    
                    # Simple approach: if any M[t] is 1, set W[t] = 1, others 0.
                    # If any M[t] is -1, set W[t] = -1, others 0.
                    # If any M[t] is 2, set W[t] = 1/2, others 0.
                    # Etc.
                    
                    found = False
                    for t in range(r):
                        if M[t] == 1:
                            W[k * n + i][t] = 1
                            found = True
                            break
                        elif M[t] == -1:
                            W[k * n + i][t] = -1
                            found = True
                            break
                        elif M[t] == 2:
                            W[k * n + i][t] = Fraction(1, 2)
                            found = True
                            break
                        elif M[t] == -2:
                            W[k * n + i][t] = Fraction(-1, 2)
                            found = True
                            break
                        elif M[t] == 3:
                            # 1/3 is not half-integer, skip
                            pass
                        elif M[t] == -3:
                            pass
                        # For other values, we'd need to combine multiple t's
                    
                    if not found:
                        # Try a linear combination
                        # This is a subset sum-like problem
                        # For now, mark as invalid
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                if verify_decomposition(U, V, W, r):
                    return U, V, W
        
        return None
    
    # Try ranks from lower bound up
    lower_bound = max(n * m, m * p, p * n)  # 20
    for r in range(lower_bound, rank_sb + 1):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best = {'U': U, 'V': V, 'W': W, 'rank': r}
            break
    
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
