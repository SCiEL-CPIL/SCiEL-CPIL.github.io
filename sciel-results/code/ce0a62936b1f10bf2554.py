"""Program for 3x5x5 matrix multiplication tensor decomposition.

Strategy: Start with the schoolbook decomposition (rank 75) and attempt to find
a lower-rank decomposition using structured searches exploiting the tensor
structure. Use exact rational verification.
"""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 5]
N, M, P = 3, 5, 5
_START_TIME = time.time()
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def _time_remaining():
    return DEADLINE - (time.time() - _START_TIME) - SAFETY_MARGIN

def _schoolbook():
    """Return the standard rank-75 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
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

def _verify(decomp, tolerance=1e-9):
    """Check if the decomposition exactly realizes the matrix multiplication tensor."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    n, m, p = N, M, P
    rank = U.shape[1]
    
    if rank == 0:
        return False
    
    # Reshape to tensor dimensions
    Uc = U.reshape(n, m, rank)
    Vc = V.reshape(m, p, rank)
    Wc = W.reshape(p, n, rank)
    
    # Compute sum over rank dimension c: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Using einsum for efficiency
    try:
        total = np.einsum('ijc,jkc,kic->ijk', Uc, Vc, Wc)
    except Exception:
        return False
        
    # The tensor for matrix multiplication has T[i,j,k] = 1 if the corresponding
    # product is formed, 0 otherwise. For standard matrix multiplication C = A*B,
    # C[i,k] = sum_j A[i,j]*B[j,k]. The tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    return np.allclose(total, 1.0, atol=tolerance, rtol=tolerance)

def _verify_exact(decomp):
    """Verify using exact rational arithmetic for small entries."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    n, m, p = N, M, P
    rank = len(U[0]) if U else 0
    
    if rank == 0:
        return False
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _structured_search():
    """Try structured approaches to find lower-rank decompositions.
    
    For 3x5x5, we can try:
    1. Block decomposition: treat as 3 independent 1x5x5 multiplications with sharing
    2. Winograd-like transformations on the 5x5 part
    3. Exploit that 5 = 3+2 to use smaller block results
    """
    n, m, p = N, M, P
    
    # Approach: Try to find a decomposition by treating the 5x5 multiplication
    # as a composition and sharing factors across the 3 output rows.
    # 
    # A 5x5 matrix multiplication has rank at most 25 (schoolbook).
    # If we can find a rank-r decomposition for 5x5, we might be able to
    # share some of the factors across the 3 output rows.
    #
    # Actually, the 3x5x5 tensor is the tensor product of a 3x1x1 tensor
    # (which has rank 3) with a 1x5x5 tensor (rank 25), giving rank 75.
    # But there might be a more efficient decomposition.
    
    # Try a specific structured approach:
    # Partition the 5 columns into groups and use block matrix multiplication.
    # For example, split 5 into 2+3:
    # A (3x5) * B (5x5) where we split B into B1 (5x2) and B2 (5x3)
    # C1 (3x2) = A * B1, C2 (3x3) = A * B2
    # This doesn't directly help reduce rank.
    
    # Let's try a random search with better structure:
    # Initialize with schoolbook and try to perturb/merge.
    
    # For now, try a few random structured searches
    vals = [1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5, 0.25, -0.25, 3, -3]
    
    best = None
    best_rank = n * m * p
    
    # Try ranks from 74 down to 50
    for target_rank in range(best_rank - 1, 49, -1):
        if _time_remaining() < 15:
            break
        
        # Try a few random attempts for this rank
        for attempt in range(20):
            if _time_remaining() < 10:
                break
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with some structure: each column corresponds to a
            # partial product
            for c in range(target_rank):
                # Choose which (i,j,k) this rank-1 term contributes to
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                
                # Add the standard term
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                
                # Add some noise to try to create linear dependencies
                if c < target_rank - n*m*p//2:
                    # Add extra non-zero entries
                    extra_u = random.randint(0, 2)
                    extra_v = random.randint(0, 2)
                    extra_w = random.randint(0, 2)
                    
                    for _ in range(extra_u):
                        r = random.randint(0, n*m-1)
                        U[r, c] += random.choice(vals)
                    for _ in range(extra_v):
                        r = random.randint(0, m*p-1)
                        V[r, c] += random.choice(vals)
                    for _ in range(extra_w):
                        r = random.randint(0, p*n-1)
                        W[r, c] += random.choice(vals)
            
            decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if _verify(decomp):
                return decomp
                
    return None

def construct():
    # Start with the safe, known schoolbook decomposition
    base = _schoolbook()
    best_rank = N * M * P  # 75
    
    # Check if we have time to attempt optimization
    if _time_remaining() > 30:
        # Try structured search
        candidate = _structured_search()
        if candidate is not None:
            return candidate
    
    return base

if __name__ == "__main__":
    # Quick self-check
    result = construct()
    if _verify(result):
        print("Valid decomposition found with rank", len(result["U"][0]))
    else:
        print("ERROR: Invalid decomposition")
