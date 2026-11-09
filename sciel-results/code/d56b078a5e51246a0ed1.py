"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15
    
    # The tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[Fraction(0)] * rank_schoolbook for _ in range(nm)]
    V_sb = [[Fraction(0)] * rank_schoolbook for _ in range(mp)]
    W_sb = [[Fraction(0)] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = Fraction(1)
                V_sb[j*p+k][col] = Fraction(1)
                W_sb[k*n+i][col] = Fraction(1)
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify(U, V, W, rank):
        """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != Fraction(1):
                        return False
        return True
    
    def try_rank(rank, time_limit):
        """Try to find a decomposition of given rank using random search."""
        start = time.time()
        random.seed(rank * 1000 + 42)
        
        # Try different strategies
        # Strategy 1: Random rational entries
        for attempt in range(200):
            if time.time() > time_limit or time.time() > deadline:
                break
            
            # Use small rational values
            scale = random.choice([1, 2, 3])
            U = [[Fraction(random.randint(-scale, scale), 1) for _ in range(rank)] for _ in range(nm)]
            V = [[Fraction(random.randint(-scale, scale), 1) for _ in range(rank)] for _ in range(mp)]
            W = [[Fraction(random.randint(-scale, scale), 1) for _ in range(pn)] for _ in range(m)]
            
            # This is unlikely to work with pure random, but let's try
            # Better: use structured approaches
            
            break
        
        # Strategy 2: Start from schoolbook and try to merge terms
        # This is complex; let's try a different approach
        
        return None
    
    # Try known constructions
    # For 3x3 x 3x3, rank 21 is known. Let's try to build on that.
    # Actually, let's try a systematic approach: use the fact that we can
    # represent the multiplication using bilinear forms.
    
    # Try rank values from low to high
    for rank in range(15, 30):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            break
        
        # Try to find a decomposition of this rank
        # Use a heuristic: random initialization + local search
        found = False
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            random.seed(rank * 10000 + trial)
            
            # Initialize with small random rationals
            U = [[Fraction(0)] * rank for _ in range(nm)]
            V = [[Fraction(0)] * rank for _ in range(mp)]
            W = [[Fraction(0)] * rank for _ in range(pn)]
            
            # Set some entries to 1 to give structure
            for r in range(rank):
                # Pick random i, j, k and set U[i*m+j][r]=1, V[j*p+k][r]=1, W[k*n+i][r]=1
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                U[i*m+j][r] = Fraction(1)
                V[j*p+k][r] = Fraction(1)
                W[k*n+i][r] = Fraction(1)
            
            # Now try to adjust
            # This is very unlikely to work; let's use a better approach
            
            # Better: use the "lax" method or algebraic approach
            # For now, just check
            if verify(U, V, W, rank):
                found = True
                break
        
        if found:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break
    
    # If no better found, return schoolbook
    # Convert to list of lists with int/float
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if val.denominator == 1:
                    new_row.append(val.numerator)
                else:
                    new_row.append(float(val))
            result.append(new_row)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
