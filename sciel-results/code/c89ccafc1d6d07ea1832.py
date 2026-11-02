"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _residual(U, V, W, n, m, p):
    """Compute the residual tensor: T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    rank = len(U[0]) if U else 0
    residual = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                residual[(i, j, k)] = 1 - total
    return residual

def _try_greedy_construct(r, n, m, p, deadline, rng):
    """
    Greedy column-by-column construction with structured initialization.
    For each column, try to maximize the number of (i,j,k) triples that are
    satisfied (residual becomes 0).
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize with zeros
    U = [[0] * r for _ in range(nm)]
    V = [[0] * r for _ in range(mp)]
    W = [[0] * r for _ in range(pn)]
    
    # Track which (i,j,k) triples are still unsatisfied
    unsatisfied = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                unsatisfied.add((i, j, k))
    
    # For each column, try to satisfy as many unsatisfied triples as possible
    for c in range(r):
        if time.time() > deadline:
            break
        if not unsatisfied:
            break
        
        # Try different patterns for this column
        best_pattern = None
        best_count = 0
        best_U_col = [0] * nm
        best_V_col = [0] * mp
        best_W_col = [0] * pn
        
        # Generate candidate patterns: small integer/half-integer entries
        # Focus on entries that are nonzero for indices relevant to unsatisfied triples
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Random pattern with small entries
            U_col = [rng.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(nm)]
            V_col = [rng.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(mp)]
            W_col = [rng.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(pn)]
            
            # Count how many unsatisfied triples this column would help with
            # A triple (i,j,k) is helped if U[i*m+j]*V[j*p+k]*W[k*n+i] != 0
            count = 0
            for (i, j, k) in unsatisfied:
                if U_col[i * m + j] != 0 and V_col[j * p + k] != 0 and W_col[k * n + i] != 0:
                    count += 1
            
            if count > best_count:
                best_count = count
                best_U_col = U_col
                best_V_col = V_col
                best_W_col = W_col
        
        if best_count == 0:
            # No improvement possible with random patterns; try structured
            # Use a pattern that targets one specific unsatisfied triple
            if unsatisfied:
                i, j, k = next(iter(unsatisfied))
                best_U_col = [0] * nm
                best_V_col = [0] * mp
                best_W_col = [0] * pn
                best_U_col[i * m + j] = 1
                best_V_col[j * p + k] = 1
                best_W_col[k * n + i] = 1
                best_count = 1
            else:
                break
        
        # Assign the best pattern to column c
        for idx in range(nm):
            U[idx][c] = best_U_col[idx]
        for idx in range(mp):
            V[idx][c] = best_V_col[idx]
        for idx in range(pn):
            W[idx][c] = best_W_col[idx]
        
        # Update unsatisfied set
        new_unsatisfied = set()
        for (i, j, k) in unsatisfied:
            total = 0
            for cc in range(c + 1):
                total += U[i * m + j][cc] * V[j * p + k][cc] * W[k * n + i][cc]
            if total != 1:
                new_unsatisfied.add((i, j, k))
        unsatisfied = new_unsatisfied
    
    # Check if we got a valid decomposition
    if not unsatisfied:
        return {"U": U, "V": V, "W": W}
    
    return None

def _try_local_search(U, V, W, n, m, p, deadline, rng):
    """
    Local search: perturb entries and check if we get closer to a valid solution.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = len(U[0])
    
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    def _score(U_, V_, W_):
        """Score = number of (i,j,k) triples where the sum equals 1"""
        count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(r):
                        total += U_[i * m + j][c] * V_[j * p + k][c] * W_[k * n + i][c]
                    if total == 1:
                        count += 1
        return count
    
    current_score = _score(best_U, best_V, best_W)
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        if current_score == n * m * p:
            break
        
        # Pick a random entry to perturb
        choice = rng.random()
        if choice < 0.33:
            mat = 'U'
            idx = rng.randint(0, nm - 1)
            col = rng.randint(0, r - 1)
            old_val = best_U[idx][col]
            new_val = old_val + rng.choice([-1, 1, -0.5, 0.5])
            best_U[idx][col] = new_val
        elif choice < 0.66:
            mat = 'V'
            idx = rng.randint(0, mp - 1)
            col = rng.randint(0, r - 1)
            old_val = best_V[idx][col]
            new_val = old_val + rng.choice([-1, 1, -0.5, 0.5])
            best_V[idx][col] = new_val
        else:
            mat = 'W'
            idx = rng.randint(0, pn - 1)
            col = rng.randint(0, r - 1)
            old_val = best_W[idx][col]
            new_val = old_val + rng.choice([-1, 1, -0.5, 0.5])
            best_W[idx][col] = new_val
        
        new_score = _score(best_U, best_V, best_W)
        
        if new_score >= current_score:
            current_score = new_score
        else:
            # Revert
            if mat == 'U':
                best_U[idx][col] = old_val
            elif mat == 'V':
                best_V[idx][col] = old_val
            else:
                best_W[idx][col] = old_val
    
    if current_score == n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return None

def _try_rank(r, n, m, p, deadline):
    """Try to find a rank-r exact decomposition using multiple strategies."""
    rng = random.Random(r * 1000 + 42)
    
    # Strategy 1: Greedy construction
    result = _try_greedy_construct(r, n, m, p, deadline, rng)
    if result is not None:
        return result
    
    # Strategy 2: Random initialization + local search
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        nm, mp, pn = n * m, m * p, p * n
        U = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
        
        result = _try_local_search(U, V, W, n, m, p, deadline, rng)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline: leave 10 seconds margin before 120s limit
    deadline = time.time() + 100
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(10, 15, 6) = 15
    for r in range(15, schoolbook_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            best = result
            break
    
    return best
# EVOLVE-BLOCK-END
