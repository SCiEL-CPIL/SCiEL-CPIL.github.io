"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # T[i,j,k] should equal 1 if this corresponds to a valid multiplication term
    # Actually, the tensor for matrix multiplication:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = delta (i.e., T[i,j,k]=1 for all i,j,k in the standard basis)
    # Wait, let me reconsider. The matrix multiplication tensor has entries:
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # because (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is all-ones of shape (n, m, p)
    # And we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Use exact arithmetic with Fractions for verification
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W):
    """Fast verification using numpy with exact rational check via integer scaling."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This is a batched operation. Let's compute it.
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    
    # Now do exact verification with fractions
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                if s != 1:
                    return False
    return True

def _search_gf3(max_rank, deadline):
    """
    Search for a rank-minimal decomposition over GF(3), then lift to rationals.
    The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k.
    We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 (mod 3)
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    n, m, p = DIMENSIONS
    num_constraints = n * m * p  # 42
    
    # We'll try to find sparse solutions over GF(3)
    # Start with lower ranks and work up
    best_result = None
    
    for rank in range(1, max_rank + 1):
        if time.time() > deadline:
            break
        if best_result is not None and rank >= len(best_result["U"][0]):
            break
        
        # Randomized search over GF(3)
        found = False
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random sparse matrices over GF(3)
            # Each entry is 0, 1, or 2 with some sparsity
            sparsity = 0.3  # probability of non-zero
            
            U_gf3 = np.zeros((n * m, rank), dtype=np.int64)
            V_gf3 = np.zeros((m * p, rank), dtype=np.int64)
            W_gf3 = np.zeros((p * n, rank), dtype=np.int64)
            
            for _ in range(n * m):
                for c in range(rank):
                    if random.random() < sparsity:
                        U_gf3[_, c] = random.randint(1, 2)
            for _ in range(m * p):
                for c in range(rank):
                    if random.random() < sparsity:
                        V_gf3[_, c] = random.randint(1, 2)
            for _ in range(p * n):
                for c in range(rank):
                    if random.random() < sparsity:
                        W_gf3[_, c] = random.randint(1, 2)
            
            # Check if this satisfies the tensor equation mod 3
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s = (s + int(U_gf3[i * m + j, c]) * int(V_gf3[j * p + k, c]) * int(W_gf3[k * n + i, c])) % 3
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Found a valid GF(3) solution!
                # Lift to rationals: interpret 0->0, 1->1, 2->-1 (or 2)
                # Use 0->0, 1->1, 2->-1 for potentially smaller values
                U_int = [[0 if x == 0 else (1 if x == 1 else -1) for x in row] for row in U_gf3.tolist()]
                V_int = [[0 if x == 0 else (1 if x == 1 else -1) for row in V_gf3.tolist()] for row in V_gf3.tolist()]
                W_int = [[0 if x == 0 else (1 if x == 1 else -1) for row in W_gf3.tolist()] for row in W_gf3.tolist()]
                
                # Verify over Q
                if _verify_fast(U_int, V_int, W_int):
                    best_result = {"U": U_int, "V": V_int, "W": W_int}
                    found = True
                    break
                else:
                    # Try alternative lifting: 2 -> 2
                    U_int2 = [[int(x) for x in row] for row in U_gf3.tolist()]
                    V_int2 = [[int(x) for x in row] for row in V_gf3.tolist()]
                    W_int2 = [[int(x) for x in row] for row in W_gf3.tolist()]
                    if _verify_fast(U_int2, V_int2, W_int2):
                        best_result = {"U": U_int2, "V": V_int2, "W": W_int2}
                        found = True
                        break
        
        if found:
            return best_result
    
    return best_result

def _search_direct(max_rank, deadline):
    """
    Direct search: try random integer matrices and verify.
    """
    n, m, p = DIMENSIONS
    best_result = None
    
    for rank in range(1, max_rank + 1):
        if time.time() > deadline:
            break
        if best_result is not None and rank >= len(best_result["U"][0]):
            break
        
        max_attempts = 3000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random sparse integer matrices
            sparsity = 0.25
            val_range = 2  # values in {-2,-1,0,1,2}
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for i in range(n * m):
                for c in range(rank):
                    if random.random() < sparsity:
                        U[i][c] = random.randint(-val_range, val_range)
            for i in range(m * p):
                for c in range(rank):
                    if random.random() < sparsity:
                        V[i][c] = random.randint(-val_range, val_range)
            for i in range(p * n):
                for c in range(rank):
                    if random.random() < sparsity:
                        W[i][c] = random.randint(-val_range, val_range)
            
            if _verify_fast(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                return best_result
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    # Get schoolbook as fallback
    schoolbook = _schoolbook()
    best_rank = len(schoolbook["U"][0])  # 42
    
    # Try to find better decomposition
    # First, try GF(3) search
    result = _search_gf3(max_rank=min(best_rank - 1, 30), deadline=deadline)
    
    if result is not None:
        return result
    
    # Try direct search
    result = _search_direct(max_rank=min(best_rank - 1, 25), deadline=deadline)
    
    if result is not None:
        return result
    
    # Return schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
