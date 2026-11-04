import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # We need to decompose the matrix multiplication tensor for 4x4 * 4x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # where A is 4x4, B is 4x7, C is 4x7
    # 
    # The tensor has entries T[i,j,k] = 1 for all valid (i,j,k)
    # A rank-1 decomposition term is u ⊗ v ⊗ w where:
    #   u is indexed by (i,j) pairs (size n*m = 16)
    #   v is indexed by (j,k) pairs (size m*p = 28)  
    #   w is indexed by (k,i) pairs (size p*n = 28)
    # 
    # We need: sum_c U[c,i,j] * V[c,j,k] * W[c,k,i] = 1 for all i,j,k
    # 
    # Strategy: Use Strassen for the 4x4 * 4x4 part (first 4 columns of B)
    # and schoolbook for the 4x4 * 4x3 part (last 3 columns of B)
    # 
    # For 4x4 * 4x4: Use Strassen's algorithm. Strassen reduces 2x2*2x2 to rank 7.
    # For 4x4*4x4, we can use a recursive Strassen approach:
    # Split 4x4 into 2x2 blocks. Each block multiplication is 2x2*2x2.
    # Standard Strassen for 4x4*4x4 uses rank 7 * (number of block multiplications)
    # Actually, the standard approach is to use Strassen recursively.
    # For 4x4*4x4, we can use 7 multiplications of 2x2*2x2 if we structure it right,
    # but actually Strassen for 4x4*4x4 typically has rank 7*4 = 28 if applied naively,
    # or we can use the fact that 4x4 can be decomposed more efficiently.
    # 
    # Actually, let's use a known result: The rank of 4x4*4x4 matrix multiplication
    # is known to be at most 49 (using Strassen recursively: 7^2 = 49).
    # But we can do better. Let me use a simpler approach first.
    # 
    # For 4x4 * 4x3, we can use schoolbook: rank = 4*4*3 = 48
    # 
    # Total rank would be: rank(4x4*4x4) + rank(4x4*4x3)
    # If we use rank 49 for 4x4*4x4 and 48 for 4x4*4x3, total = 97
    # 
    # But wait, we can potentially share some terms. However, for simplicity,
    # let's first implement Strassen for 4x4*4x4 and schoolbook for 4x4*4x3.
    # 
    # Actually, let me reconsider. The simplest approach that definitely works:
    # Use schoolbook for the whole thing: rank 112.
    # 
    # Let me try to use Strassen for the 4x4*4x4 part.
    # Strassen's algorithm for 2x2*2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # 
    # For 4x4*4x4, we split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # 
    # C = A*B, so:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # 
    # We can apply Strassen to each of these 2x2*2x2 multiplications, but that
    # would give 4 * 7 = 28 rank terms.
    # 
    # Alternatively, we can use a more clever approach. But for now, let's
    # implement the 28-term Strassen for 4x4*4x4 and 48-term schoolbook for 4x4*4x3.
    # Total rank: 28 + 48 = 76
    # 
    # Wait, that's better than 112! Let me implement this.
    
    terms = []
    
    def make_u(idx_i, idx_j, val):
        u = [0] * (n * m)
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val):
        v = [0] * (m * p)
        v[idx_j * p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val):
        w = [0] * (p * n)
        w[idx_k * n + idx_i] = val
        return w
    
    # For 4x4 * 4x4 (first 4 columns of B), use Strassen
    # We'll handle the first 4 columns (k=0,1,2,3)
    
    # Strassen for 2x2*2x2 blocks
    # For each block multiplication, we have 7 terms
    # We need to map these to the full tensor decomposition
    
    # Let's define the Strassen terms for a 2x2*2x2 multiplication
    # where the input blocks are at positions (i_offset, j_offset) for A
    # and (j_offset, k_offset) for B, output at (i_offset, k_offset)
    
    def strassen_terms(i_off, j_off, k_off):
        """Generate Strassen terms for 2x2*2x2 block multiplication"""
        result = []
        
        # M1 = (A11+A22)*(B11+B22)
        # A11 is at (i_off, j_off), A22 is at (i_off+1, j_off+1)
        # B11 is at (j_off, k_off), B22 is at (j_off+1, k_off+1)
        # Output C11 += M1, C22 += M1
        
        # Term 1: M1 contributes to C11 and C22
        u1 = [0] * (n * m)
        u1[(i_off) * m + j_off] = 1
        u1[(i_off+1) * m + (j_off+1)] = 1
        v1 = [0] * (m * p)
        v1[(j_off) * p + k_off] = 1
        v1[(j_off+1) * p + (k_off+1)] = 1
        w1 = [0] * (p * n)
        w1[(k_off) * n + i_off] = 1
        w1[(k_off+1) * n + (i_off+1)] = 1
        result.append((u1, v1, w1))
        
        # M2 = (A21+A22)*B11
        # A21 is at (i_off+1, j_off), A22 is at (i_off+1, j_off+1)
        # B11 is at (j_off, k_off)
        # Output C21 += M2, C22 -= M2
        
        u2 = [0] * (n * m)
        u2[(i_off+1) * m + j_off] = 1
        u2[(i_off+1) * m + (j_off+1)] = 1
        v2 = [0] * (m * p)
        v2[(j_off) * p + k_off] = 1
        w2 = [0] * (p * n)
        w2[(k_off) * n + (i_off+1)] = 1
        # M2 contributes +1 to C21 and -1 to C22
        # But we need separate terms for each output
        # Actually, in the tensor decomposition, each term is a rank-1 tensor
        # that contributes to all (i,j,k) triples where the product is nonzero
        # So we need to be more careful
        
        # Let me reconsider. The Strassen algorithm gives:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # 
        # Each M is a product of two linear combinations.
        # In the tensor decomposition, each rank-1 term corresponds to one
        # "multiplication" in the algorithm.
        
        # For M1 = (A11+A22)*(B11+B22):
        # This contributes to C11 with +1 and to C22 with +1
        # So the rank-1 term has:
        # u: A11=1, A22=1
        # v: B11=1, B22=1
        # w: C11=1, C22=1
        
        # For M2 = (A21+A22)*B11:
        # This contributes to C21 with +1 and to C22 with -1
        # u: A21=1, A22=1
        # v: B11=1
        # w: C21=1, C22=-1
        
        # And so on...
        
        # Let me redo this properly.
        
        # Clear the result and redo
        result = []
        
        # M1 = (A11+A22)*(B11+B22)
        # Contributes: C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M2 = (A21+A22)*B11
        # Contributes: C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        result.append((u, v, w))
        
        # M3 = A11*(B12-B22)
        # Contributes: C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M4 = A22*(B21-B11)
        # Contributes: C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M5 = (A11+A12)*B22
        # Contributes: C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        result.append((u, v, w))
        
        # M6 = (A21-A11)*(B11+B12)
        # Contributes: C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M7 = (A12-A22)*(B21+B22)
        # Contributes: C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        result.append((u, v, w))
        
        return result
    
    # Apply Strassen to each 2x2*2x2 block in the 4x4*4x4 multiplication
    # The 4x4 matrix is divided into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # 
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # 
    # Each of these is a sum of two 2x2*2x2 multiplications.
    # We can apply Strassen to each 2x2*2x2 multiplication.
    
    # For C11 = A11*B11 + A12*B21:
    # A11*B11: i_off=0, j_off=0, k_off=0
    # A12*B21: i_off=0, j_off=2, k_off=0 (but B21 has j_off=2, k_off=0)
    # Wait, I need to be more careful with the indexing.
    
    # Let me define the block positions:
    # A blocks:
    # A11: rows 0-1, cols 0-1 -> i_off=0, j_off=0
    # A12: rows 0-1, cols 2-3 -> i_off=0, j_off=2
    # A21: rows 2-3, cols 0-1 -> i_off=2, j_off=0
    # A22: rows 2-3, cols 2-3 -> i_off=2, j_off=2
    # 
    # B blocks:
    # B11: rows 0-1, cols 0-1 -> j_off=0, k_off=0
    # B12: rows 0-1, cols 2-3 -> j_off=0, k_off=2
    # B21: rows 2-3, cols 0-1 -> j_off=2, k_off=0
    # B22: rows 2-3, cols 2-3 -> j_off=2, k_off=2
    # 
    # C blocks:
    # C11: rows 0-1, cols 0-1 -> i_off=0, k_off=0
    # C12: rows 0-1, cols 2-3 -> i_off=0, k_off=2
    # C21: rows 2-3, cols 0-1 -> i_off=2, k_off=0
    # C22: rows 2-3, cols 2-3 -> i_off=2, k_off=2
    
    # For each block multiplication, we need to apply Strassen
    # But Strassen is for 2x2*2x2, so we apply it to each block pair
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: use Strassen with i_off=0, j_off=0, k_off=0
    terms.extend(strassen_terms(0, 0, 0))
    # A12*B21: use Strassen with i_off=0, j_off=2, k_off=0
    # But wait, B21 is at j_off=2, k_off=0
    terms.extend(strassen_terms(0, 2, 0))
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: use Strassen with i_off=0, j_off=0, k_off=2
    terms.extend(strassen_terms(0, 0, 2))
    # A12*B22: use Strassen with i_off=0, j_off=2, k_off=2
    terms.extend(strassen_terms(0, 2, 2))
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: use Strassen with i_off=2, j_off=0, k_off=0
    terms.extend(strassen_terms(2, 0, 0))
    # A22*B21: use Strassen with i_off=2, j_off=2, k_off=0
    terms.extend(strassen_terms(2, 2, 0))
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: use Strassen with i_off=2, j_off=0, k_off=2
    terms.extend(strassen_terms(2, 0, 2))
    # A22*B22: use Strassen with i_off=2, j_off=2, k_off=2
    terms.extend(strassen_terms(2, 2, 2))
    
    # That gives 8 * 7 = 56 terms for 4x4*4x4
    # But this is not optimal. The standard Strassen for 4x4*4x4 should use
    # 7^2 = 49 terms if done recursively, but my approach above uses 56.
    # 
    # Actually, the issue is that I'm applying Strassen to each 2x2*2x2 block
    # multiplication separately, which gives 8 block multiplications * 7 terms = 56.
    # But the correct recursive Strassen would share some computations.
    # 
    # For now, let's use this 56-term approach and add schoolbook for 4x4*4x3.
    
    # For 4x4 * 4x3 (last 3 columns of B, k=4,5,6), use schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # k = 4, 5, 6
                u = make_u(i, j, 1)
                v = make_v(j, k, 1)
                w = make_w(k, i, 1)
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
