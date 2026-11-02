"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook rank n*m*p."""
    r = n * m * p
    U = [[0.0] * r for _ in range(n * m)]
    V = [[0.0] * r for _ in range(m * p)]
    W = [[0.0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W decompose the matrix multiplication tensor."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification using numpy."""
    r = len(U[0])
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i * m + j], Va[j * p + k]) * Wa[k * n + i]
                # Actually need elementwise: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                # That's not a simple dot product. Let me fix.
    # Proper vectorized check:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_strassen_block(n, m, p):
    """
    Try to use Strassen on 2x2 sub-blocks to reduce rank.
    Partition A(4x5) into rows [0:2, 2:4], cols [0:2, 2:5]
    Partition B(5x5) into rows [0:2, 2:5], cols [0:2, 2:5]
    C(4x5) blocks: C11(2x2), C12(2x3), C21(2x2), C22(2x3)
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    For C11: A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    For C21: A21(2x2)*B11(2x2) + A22(2x3)*B21(3x2)
    
    We can apply Strassen to the 2x2*2x2 parts:
    C11 = A11*B11 + A12*B21
    C21 = A21*B11 + A22*B21
    
    Actually, let me think about this more carefully.
    
    The full product C = A*B where A is 4x5, B is 5x5.
    
    Let me partition differently. Split A into two 2x5 blocks (top/bottom),
    and B into 5x2 and 5x3 blocks (left/right).
    
    C_top(2x5) = A_top(2x5) * B(5x5)
    C_bot(2x5) = A_bot(2x5) * B(5x5)
    
    This doesn't help directly. Let me try a different approach.
    
    Actually, for 4x5 * 5x5, the schoolbook rank is 100.
    The best known upper bound for 4x5*5x5 might be lower.
    
    Let me try: partition into 2x2 and 2x3 blocks.
    A = [A1 A2] where A1 is 2x2, A2 is 2x3 (top row block, split cols)
       [A3 A4] where A3 is 2x2, A4 is 2x3 (bottom row block, split cols)
    B = [B1 B2] where B1 is 2x2, B2 is 2x3 (top row block of B, split cols)
       [B3 B4] where B3 is 3x2, B4 is 3x3 (bottom row block of B, split cols)
    
    Wait, B is 5x5. Split rows into 2+3, cols into 2+3.
    B11(2x2), B12(2x3)
    B21(3x2), B22(3x3)
    
    A is 4x5. Split rows into 2+2, cols into 2+3.
    A11(2x2), A12(2x3)
    A21(2x2), A22(2x3)
    
    C11(2x2) = A11*B11 + A12*B21
    C12(2x3) = A11*B12 + A12*B22
    C21(2x2) = A21*B11 + A22*B21
    C22(2x3) = A21*B12 + A22*B22
    
    Now for C11: A11(2x2)*B11(2x2) uses 4 schoolbook terms, or 7 Strassen terms.
    A12(2x3)*B21(3x2) uses 6 schoolbook terms (2*3*2 = 12? No, 2*3*2=12 terms for full schoolbook of 2x3*3x2)
    
    Wait, 2x3 * 3x2 schoolbook has rank 2*3*2 = 12. That's worse.
    
    Hmm, let me reconsider. The total schoolbook rank is 4*5*5 = 100.
    
    For the block decomposition:
    C11: 2*2*2 + 2*3*2 = 8 + 12 = 20
    C12: 2*2*3 + 2*3*3 = 12 + 18 = 30
    C21: 2*2*2 + 2*3*2 = 8 + 12 = 20
    C22: 2*2*3 + 2*3*3 = 12 + 18 = 30
    Total: 20+30+20+30 = 100. Same as schoolbook.
    
    To reduce rank, I need to use Strassen on the 2x2*2x2 parts.
    A11*B11: 4 terms -> 7 terms (Strassen). Net change: +3
    A21*B11: 4 terms -> 7 terms (Strassen). Net change: +3
    That makes it worse!
    
    What if I combine the Strassen terms?
    A11*B11 and A21*B11 both involve B11.
    
    Let me try a different partition. What about 4x5*5x5 where I use
    the fact that 4 = 2+2, 5 = 2+3?
    
    Actually, the known lower bound for 4x5*5x5 is at least... let me think.
    The minimum rank for matrix multiplication m x n * n x p is at least max(m*n, n*p, m*p) = max(20, 25, 20) = 25.
    But that's a very loose bound.
    
    For now, let me just try to find any decomposition with rank < 100 using a more systematic approach.
    
    One idea: use the fact that some entries can be shared.
    
    Actually, let me try a completely different approach. Use the "rank 1" decomposition
    where each term picks one (i,j,k) triple, but try to merge terms where possible.
    
    For the matrix multiplication tensor T[i,j,k] = 1, a rank-1 term is U[i,j]*V[j,k]*W[k,i]
    which must equal 1 for the specific (i,j,k) and 0 for all other triples.
    
    This is essentially what schoolbook does. To reduce rank, we need non-trivial
    combinations.
    
    Let me try a random search with a smarter initialization.
    """
    pass


def _try_random_search(n, m, p, deadline):
    """Try random search for low-rank decompositions."""
    best_rank = n * m * p
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best = (best_rank, U_sb, V_sb, W_sb)
    
    # Try ranks from 99 down
    for target_rank in range(99, 24, -1):
        if time.time() > deadline - 10:
            break
        
        # Limit attempts based on remaining time
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Use small integer values for better chance of exact solutions
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify(U, V, W, n, m, p):
                best = (target_rank, U, V, W)
                break
        
        if best[0] < best_rank:
            best_rank = best[0]
            break
    
    return best


def _try_structured(n, m, p, deadline):
    """
    Try a structured approach: start with schoolbook and try to merge terms.
    Or try known algorithms like Strassen variants.
    """
    best_rank = n * m * p
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best = (best_rank, U_sb, V_sb, W_sb)
    
    # Try to find a rank-99 decomposition by modifying one schoolbook term
    # This is unlikely to work, but let's try a more principled approach.
    
    # Use numpy to set up the tensor and try to find low-rank approximations
    # that are exact.
    
    # The tensor T is n*m*p = 100 dimensional (flattened).
    # We want to find U, V, W with rank r such that the tensor equation holds.
    
    # This is a system of 100 polynomial equations in r*(n*m + m*p + p*n) variables.
    # For r < 100, this is overdetermined but may have solutions.
    
    # Let me try a different structured approach: use the "border rank" concept
    # and try to find exact decompositions with slightly lower rank.
    
    # For now, return schoolbook if no better found.
    return best


def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 130  # Leave 20s margin
    
    # Start with schoolbook
    best_rank = n * m * p
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best = (best_rank, U_sb, V_sb, W_sb)
    
    # Try random search for better rank
    if time.time() < deadline - 30:
        result = _try_random_search(n, m, p, deadline)
        if result[0] < best_rank:
            best = result
            best_rank = result[0]
    
    # Try structured approach
    if time.time() < deadline - 10:
        result = _try_structured(n, m, p, deadline)
        if result[0] < best_rank:
            best = result
            best_rank = result[0]
    
    r, U, V, W = best
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
