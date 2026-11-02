"""Find rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the target tensor T[i,j,k] = 1 for all (i,j,k)
    # Actually, the tensor is T[i,j,k,l] = delta_{j,l} for matrix mult
    # But in the factorization form: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # The "tensor" we're decomposing has entries T[i,j,k] = 1 if the (i,j) of A and (j,k) of B multiply
    # More precisely, the bilinear map: (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # The trilinear form is: sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k)
    
    # We need: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for "wrong" combinations... actually the tensor only has 1s at valid positions
    
    # Wait, I need to be more careful. The matrix multiplication tensor T has:
    # T[i,j,k] = 1 if j is a valid index (which it always is for 0<=j<m)
    # Actually no. The tensor T_{i,j,k} for matrix mult A(m x n) * B(n x p) is:
    # T_{i,j,k} = 1 if the j-th column of A and j-th row of B are multiplied
    # In our case: A is 2x3, B is 3x9, so j ranges 0..2
    # T[i,j,k] = 1 for all i in 0..1, j in 0..2, k in 0..8
    
    # So we need: sum_c U[i*3+j, c] * V[j*9+k, c] * W[k*2+i, c] = 1 for all (i,j,k)
    
    # Let's try to find a low-rank decomposition
    # Start with schoolbook and try to reduce
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Strategy: try rank r from best_rank-1 downward, search for solutions
    
    # For small ranks, we can try random initialization + local search
    # Or use the fact that we need integer/half-integer entries
    
    # Let's try a few approaches:
    # 1. Try rank = 53, 52, ... with random sparse integer vectors
    # 2. Use structured patterns
    
    # Given time constraints, let's try a limited search
    # Try to see if we can do better than schoolbook
    
    # One idea: can we combine two terms? E.g., if we have:
    # U[a,c1]*V[b,c1]*W[d,c1] + U[a,c2]*V[b,c2]*W[d,c2] = 1
    # and similarly for other (i,j,k)
    
    # This is complex. Let's try a numerical approach first to see what's achievable.
    
    # Try rank values from 53 down to, say, 40
    # For each rank, try random integer vectors and check
    
    # Actually, let's try a smarter approach: use the structure
    # The tensor has 2*3*9 = 54 entries, all equal to 1
    # We need to express each as a sum of products
    
    # Let me try: can we use rank 53? That would mean we can merge two schoolbook terms.
    # For that, we'd need: for some (i1,j1,k1) and (i2,j2,k2),
    # U[i1*3+j1, c]*V[j1*9+k1, c]*W[k1*2+i1, c] + 
    # U[i2*3+j2, c]*V[j2*9+k2, c]*W[k2*2+i2, c] = 1 for the merged term
    # while maintaining all other equations
    
    # This is very constrained. Let's just try random search for a few ranks.
    
    def try_random_rank(rank, num_trials=10000):
        """Try to find a decomposition of given rank using random integer/half-integer vectors."""
        if time.time() > deadline - 10:
            return None
        
        # Try random sparse integer vectors
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Initialize with random small integers
            for c in range(rank):
                # Each column should have some non-zero entries
                # Try to make it work by random assignment
                U[:, c] = np.random.choice([-2, -1, 0, 0, 0, 1, 2], size=n*m)
                V[:, c] = np.random.choice([-2, -1, 0, 0, 0, 1, 2], size=m*p)
                W[:, c] = np.random.choice([-2, -1, 0, 0, 0, 1, 2], size=p*n)
            
            # Check if this works (unlikely to work randomly)
            if verify(U.tolist(), V.tolist(), W.tolist(), rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try a few lower ranks
    for rank in range(schoolbook_rank - 1, max(schoolbook_rank - 10, 30), -1):
        if time.time() > deadline - 20:
            break
        
        result = try_random_rank(rank, num_trials=5000)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
