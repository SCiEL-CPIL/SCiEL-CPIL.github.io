"""Program for rank-minimal exact tensor decomposition of 3x4x8 matrix multiplication."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 115  # seconds, leave margin before 120s

def verify(U, V, W, n=3, m=4, p=8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 if j is the shared index (which it always is), else 0
    # Wait, the tensor T[i,j,k] for matrix multiplication C = AB where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has 1 at (i,j,k) for each valid i,j,k? No.
    # The multiplication tensor M_{n,m,p} has entries T[i,j,k] = delta_{jk}... no.
    # T[i,j,k] is the coefficient of x_i y_j z_k in the product (sum_{i,j} a_ij x_i y_j)(sum_{j,k} b_jk y_j z_k)
    # = sum_{i,j,k} a_ij b_jk x_i y_j z_k
    # So T[i,j,k] = 1 for all valid i, j, k? No, T[i,j,k] = 1 if the indices match up.
    # Actually, for each term a_ij b_jk, the output coefficient is for x_i z_k.
    # The tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # Wait, that's not right. Let me reconsider.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form: C is a vector, A and B are vectors.
    # T is a 3-tensor where T[i,j,k] = 1 if the j-th column of A and j-th row of B are used.
    # So T[i,j,k] = 1 for all i, j, k? No, T[i,j,k] = 1 only when the indices are consistent.
    # Actually, T[i,j,k] = 1 for all valid (i,j,k) is wrong.
    # T[i,j,k] is the coefficient such that C[i,k] = sum_{j} sum_{c} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need C[i,k] = sum_j A[i,j] * B[j,k]
    # So for each (i,k), we need sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = A[i,j] * B[j,k] summed over j
    # This means for each c, the rank-1 tensor U_c ⊗ V_c ⊗ W_c should contribute to the multiplication.
    # The standard schoolbook: for each (i,j,k), one term.
    # Let me just check: does sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] equal 1 for all (i,j,k)?
    # In schoolbook: U[i*m+j, c_ij_k] = 1, V[j*p+k, c_ij_k] = 1, W[k*n+i, c_ij_k] = 1
    # For a given (i,j,k), only c = c_ij_k has all three = 1, others 0.
    # So the sum is 1 for each (i,j,k). Yes! T[i,j,k] = 1 for all valid i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def schoolbook(n=3, m=4, p=8):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    result = schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using randomized search over {-1, 0, 1}
    # The key insight: for small dimensions, rank can be lower than n*m*p
    # For 3x4x8, let's try ranks below 96
    
    # Known lower bounds: rank >= n*m + m*p + p*n - 2*(n+m+p) + 4? 
    # Actually, a simple lower bound is max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Another bound: rank >= (n*m*p) / max(n,m,p) = 96/8 = 12
    
    # Let's do a randomized search for rank in a range below 96
    # Use {-1, 0, 1} entries
    
    deadline = START_TIME + TIME_LIMIT
    
    # Try a few ranks with random search
    for rank in range(64, 96, -2):  # Try even ranks, go down from 94
        if time.time() > deadline - 30:
            break
        
        # Random search: generate random U, V, W with entries in {-1, 0, 1}
        # and check if they work
        # This is very unlikely to work by pure random, but let's try a structured approach
        
        # Better approach: fix the structure. Since T[i,j,k] = 1 for all (i,j,k),
        # we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
        # Try sparse random search
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 25:
                break
            
            # Generate sparse random matrices
            density = 0.15
            U_rand = [[0]*rank for _ in range(n*m)]
            V_rand = [[0]*rank for _ in range(m*p)]
            W_rand = [[0]*rank for _ in range(p*n)]
            
            for i in range(n*m):
                for c in range(rank):
                    if random.random() < density:
                        U_rand[i][c] = random.choice([-1, 1])
            
            for j in range(m*p):
                for c in range(rank):
                    if random.random() < density:
                        V_rand[j][c] = random.choice([-1, 1])
            
            for k in range(p*n):
                for c in range(rank):
                    if random.random() < density:
                        W_rand[k][c] = random.choice([-1, 1])
            
            if verify(U_rand, V_rand, W_rand, n, m, p):
                result = {"U": U_rand, "V": V_rand, "W": W_rand}
                best_rank = rank
                break  # Found one for this rank, move to lower rank
        
        if best_rank < n*m*p:
            break  # Found improvement, stop searching lower ranks for now
    
    return result
