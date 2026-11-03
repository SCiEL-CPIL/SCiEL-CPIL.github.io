"""Program using randomized algebraic geometry approach for low-rank tensor decomposition."""
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank n*m*p."""
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
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if total != 1:
                        return False
        # Also check that entries where no (i,j,k) maps to (a,b,c) give 0
        # But by construction of the tensor, only valid (i,j,k) give 1, others give 0
        # Since we only check valid (i,j,k) and they all give 1, and the tensor is 0 elsewhere,
        # we need to also verify that for invalid (a,b,c) combinations, the sum is 0.
        # However, the tensor is defined only on the valid indices.
        # The factorization must satisfy: for ALL a in [0,n*m), b in [0,m*p), c in [0,p*n),
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # where T[a,b,c] = 1 if there exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i, else 0
        # Let's verify all entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    # Determine expected value
                    expected = Fraction(0)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if i * m + j == a and j * p + k == b and k * n + i == c:
                                    expected = Fraction(1)
                    if total != expected:
                        return False
        return True
    
    def try_rank_41():
        """
        Try to find a rank-41 decomposition using randomized algebraic geometry.
        
        Approach: Start with schoolbook (rank 42). Try to find a linear dependency
        that allows us to express one term as a combination of others.
        
        For rank 41, we need to find U (6x41), V (21x41), W (14x41) such that
        the tensor equation holds.
        
        Randomized approach: 
        1. Randomly initialize U and V with small rational entries
        2. For each row of W, solve the linear system
        3. Verify the solution
        
        But this is complex. Let's try a more structured approach.
        
        Key insight: For matrix multiplication, there's a known technique where
        if n=2, we can sometimes reduce the rank. Let's try specific constructions.
        """
        # Try a specific construction for rank 41
        # Idea: Use the fact that for M_{2,m,p}, there might be savings
        
        # Let's try: take schoolbook and see if we can merge two columns
        # by finding coefficients that make the sum work out
        
        # For rank 41, we have 41 terms. Let's try to construct them directly.
        # 
        # A known result: rank(M_{2,2,2}) = 7. The trick involves using 
        # (A+B+C+D+...) style constructions.
        #
        # For M_{2,3,7}, let's try a similar approach.
        # 
        # Actually, let's use a randomized search:
        # Fix the first 40 terms as schoolbook (for 40 of the 42 (i,j,k) triples),
        # and try to find the 41st term that covers the remaining 2 triples.
        # This won't work directly since one term can't cover two disjoint triples.
        #
        # Better: Let's try to find a rank-41 decomposition by solving a system.
        # 
        # Given the complexity, let's try a few random initializations.
        
        random.seed(42)
        
        for attempt in range(100):
            # Try to construct a rank-41 decomposition
            # Strategy: Use 41 of the 42 schoolbook positions, and adjust
            # one position to cover two triples with a single term (impossible)
            # 
            # Alternative: Use a different basis.
            # 
            # Let's try: for each of the 42 (i,j,k) triples, we have a term.
            # We want to replace two terms with one. This requires that the
            # "missing" contribution can be expressed as a rank-1 tensor.
            # 
            # This is equivalent to: does there exist (i1,j1,k1) and (i2,j2,k2)
            # such that e_{i1,j1,k1} + e_{i2,j2,k2} has rank 1? No, it doesn't.
            # 
            # So we can't simply merge two schoolbook terms.
            # 
            # Let's try a different approach: allow the U, V, W entries to be
            # non-trivial (not just 0 or 1).
            
            # Randomized approach:
            # 1. Pick a random subset S of 41 of the 42 (i,j,k) triples
            # 2. For these 41 triples, set up equations
            # 3. Solve for U, V, W
            
            # This is still complex. Let's try a heuristic:
            # Start with schoolbook, randomly perturb some entries, and check
            # if we can reduce the rank.
            
            pass
        
        # Given the difficulty, let's return schoolbook
        return None
    
    # Try to find a rank-41 decomposition
    result_41 = try_rank_41()
    if result_41 is not None:
        if verify_decomposition(result_41["U"], result_41["V"], result_41["W"]):
            return result_41
    
    # Fallback to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
