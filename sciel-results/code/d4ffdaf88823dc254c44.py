"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 105  # Safety margin before 120s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition using structured approach
    # For 3x3 * 3x8, we can try to use Strassen-like techniques
    
    # The key insight: for matrix multiplication C = A * B where A is n x m, B is m x p
    # The tensor T has entries T_{(i,j),(j,k),(k,i)} = 1
    # We want to find rank-r decomposition: T = sum_r U_{(i,j),r} V_{(j,k),r} W_{(k,i),r}
    
    # For m=3, we have 3 "layers" each of size n x p = 3 x 8 = 24
    # Schoolbook uses 72 terms
    
    # Let's try a random search approach with verification
    # We'll try ranks from 24 (lower bound) up to 72
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reproduces the multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The tensor entry T_{(i,j),(j,k),(k,i)} should be 1
                    # All other entries should be 0
                    
                    # Compute the reconstructed value for position (i,j,k)
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    
                    if val != 1:
                        return False
        
        # Check that non-multiplication entries are zero
        # For each (i,j,k), the tensor entry is 1 if j is the connecting index
        # Actually, we need to check that for each triple (i,j,k), the only non-zero
        # contribution comes from valid multiplication terms
        
        # More carefully: the tensor M_{n,m,p} has entries:
        # M_{(i,j),(j,k),(k,i)} = 1 for all valid i,j,k
        # All other entries are 0
        
        # In our factorization, the (a,b,c) entry of the tensor is:
        # sum_r U[a,r] * V[b,r] * W[c,r]
        # where a = i*m+j, b = j'*p+k', c = k''*n+i''
        
        # For this to equal the multiplication tensor, we need:
        # sum_r U[i*m+j,r] * V[j'*p+k',r] * W[k''*n+i'',r] = 1 if j=j', k=k'', i=i''
        # and 0 otherwise
        
        # Let's verify this more carefully
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    for j2 in range(m):
                        for k2 in range(p):
                            b = j2 * p + k2
                            for i2 in range(n):
                                c = k2 * n + i2  # Wait, this is wrong
                                # c should be k'*n + i' where k' and i' are the indices
                                # Let me reconsider the indexing
                                
        # Actually, let's just verify by checking the full tensor reconstruction
        # The tensor has dimensions (n*m) x (m*p) x (p*n)
        # Entry (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i should be 1
        # All other entries should be 0
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Compute the reconstructed value
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    
                    # Determine what the true tensor entry should be
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    
                    # The true entry is 1 if j == j2 and k == k2 and i == i2
                    # Wait, that's not right either. Let me think again.
                    
                    # The multiplication tensor M_{n,m,p} satisfies:
                    # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
                    # So the tensor entry T_{(i,j),(j,k),(k,i)} = 1
                    # This means: for indices (a,b,c) = (i*m+j, j*p+k, k*n+i), T=1
                    # For all other combinations, T=0
                    
                    # So we need: j == j2 AND k == k2 AND i == i2
                    # But wait, that's not quite right. Let's be more careful.
                    
                    # a = i*m + j, so i = a//m, j = a%m
                    # b = j*p + k, so j2 = b//p, k = b%p
                    # c = k*n + i, so k2 = c//n, i2 = c%n
                    
                    # The tensor entry is 1 if and only if:
                    # j == j2 (the middle index matches)
                    # AND k == k2 (the output column matches)
                    # AND i == i2 (the output row matches)
                    
                    if j == j2 and k == k2 and i == i2:
                        expected = 1
                    else:
                        expected = 0
                    
                    if val != expected:
                        return False
        
        return True
    
    # Try random search for lower ranks
    # We'll use a simple approach: generate random factor matrices and verify
    
    # For efficiency, let's try a few structured approaches first
    
    # Approach: Use the fact that for each j, we have a rank-1 tensor
    # T_j = e_i * e_k (outer product) for the j-th layer
    # We can try to combine these across j values
    
    # For m=3, we have 3 layers. Can we do better than 3*24=72?
    
    # Let's try a random search with small ranks
    for target_rank in range(24, min(72, 50)):
        if time.time() > deadline - 10:
            break
        
        # Try a few random attempts for this rank
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Generate random factor matrices with small integer entries
            # Use a sparse approach: most entries are 0
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # For each term r, choose random (i,j,k) and set the corresponding entries
            for r in range(target_rank):
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                
                # Use small random coefficients
                coeff = random.choice([-2, -1, 1, 2])
                
                U[i * m + j][r] = coeff
                V[j * p + k][r] = coeff
                W[k * n + i][r] = coeff
            
            # Verify
            if verify_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if best_rank < schoolbook_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
