"""Rewritten program: search for a rank-minimal exact tensor decomposition of 2x3 * 3x7."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
DEADLINE = 110.0  # safety margin under 120s limit (and 150s kill)


def _verify(U, V, W):
    """Check exact tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    Returns True if all 2*3*7 entries match the identity tensor (1 if i,j,k all 'aligned' else 0).
    For matrix mult 2x3 * 3x7, the tensor T[i,j,k] is 1 if the multiplication contributes,
    i.e., T[i,j,k] = 1 for all (i,j,k) in the schoolbook sense? No.
    
    Actually, the matrix multiplication tensor T has T[i,j,k] = 1 if we multiply A[i,j]*B[j,k].
    So the tensor is the standard matmul tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Wait, that's not right either. The tensor for matrix multiplication M(n,m,p) is defined as:
    T[i,j,k] = 1 if the (i,j,k) entry corresponds to A[i,j]*B[j,k]. So T[i,j,k] = 1 for all valid i,j,k.
    
    So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(len(U[0])):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True


def _verify_fast(U, V, W):
    """Faster verification using integer arithmetic (multiply by 2 to avoid fractions if needed)."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                ui = U[i*m+j]
                vj = V[j*p+k]
                wk = W[k*n+i]
                for c in range(rank):
                    total += ui[c] * vj[c] * wk[c]
                if total != 1:
                    return False
    return True


def _try_rank_bruteforce(rank, max_iters=5000, timeout=10.0):
    """Try to find a rank-r decomposition by random search with integer entries in {-1, 0, 1}.
    Returns (U, V, W) or None.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    
    for it in range(max_iters):
        if time.time() - start > timeout:
            break
        
        # Random sparse matrices with entries in {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
        
        if _verify_fast(U, V, W):
            return U, V, W
    
    return None


def _try_rank_greedy(rank, max_iters=2000, timeout=10.0):
    """Greedy/random approach: start from schoolbook and try to merge columns."""
    n, m, p = DIMENSIONS
    # Start with schoolbook
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Initialize with first 'rank' schoolbook columns
    for c in range(rank):
        i = c // (m*p)
        rem = c % (m*p)
        j = rem // p
        k = rem % p
        U[i*m+j][c] = 1
        V[j*p+k][c] = 1
        W[k*n+i][c] = 1
    
    # Try to find a better solution by random perturbation
    start = time.time()
    best = None
    for it in range(max_iters):
        if time.time() - start > timeout:
            break
        
        # Randomly modify entries
        U2 = [row[:] for row in U]
        V2 = [row[:] for row in V]
        W2 = [row[:] for row in W]
        
        # Perturb a few entries
        for _ in range(3):
            mat = random.choice([U2, V2, W2])
            r = random.randint(0, len(mat)-1)
            c = random.randint(0, rank-1)
            mat[r][c] = random.choice([-1, 0, 1])
        
        if _verify_fast(U2, V2, W2):
            return U2, V2, W2
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x7."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank is n*m*p = 42
    schoolbook_rank = n * m * p
    
    # Create schoolbook decomposition
    U = [[0]*schoolbook_rank for _ in range(n*m)]
    V = [[0]*schoolbook_rank for _ in range(m*p)]
    W = [[0]*schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Verify schoolbook
    assert _verify_fast(U, V, W), "Schoolbook verification failed"
    
    best_rank = schoolbook_rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank solutions
    # Known lower bounds: for M(2,3,7), the rank is at least... 
    # Let's try ranks from 30 down to schoolbook_rank
    # Actually, let's be more strategic. Try a few specific ranks.
    
    # For small matrix multiplication, known results:
    # M(2,2,2) has rank 7 (Strassen)
    # M(2,3,3) has rank... let's just search
    
    # Try ranks in descending order from schoolbook - 1
    for rank in range(schoolbook_rank - 1, 20, -1):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Quick random search
        result = _try_rank_bruteforce(rank, max_iters=2000, timeout=5.0)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            # Continue searching for even lower rank
            continue
    
    # If we found a solution, also try to improve further
    # But for now, return the best found
    
    return best_result


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
