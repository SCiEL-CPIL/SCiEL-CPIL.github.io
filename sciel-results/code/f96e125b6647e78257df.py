"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i*m+j, j*p+k, k*n+i] for valid indices
    # Actually, let's verify by checking that for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for all other combinations, the sum should be 0.
    
    # The tensor T has T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k
    # More precisely: T[(i,k), (i,j), (j,k)] = 1
    
    # Let's check all triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j  # index in first input
                b = j * p + k  # index in second input  
                c = k * n + i  # index in output
                val = 0
                for col in range(rank):
                    val += U[a][col] * V[b][col] * W[c][col]
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check that off-diagonal terms are zero
    # For any (a, b, c) where there's no valid (i,j,k) such that a=i*m+j, b=j*p+k, c=k*n+i
    # This is complex, so let's check a sample
    # Actually, the full check requires checking all 12*6*18 = 1296 entries
    # Let's do it properly
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Determine if this is a valid tensor entry
                # a = i*m + j => i = a // m, j = a % m
                i_a = a // m
                j_a = a % m
                # b = j*p + k => j = b // p, k = b % p
                j_b = b // p
                k_b = b % p
                # c = k*n + i => k = c // n, i = c % n
                k_c = c // n
                i_c = c % n
                
                is_valid = (j_a == j_b) and (i_a == i_c) and (k_b == k_c)
                
                val = 0
                for col in range(rank):
                    val += U[a][col] * V[b][col] * W[c][col]
                
                if is_valid:
                    if abs(val - 1) > 1e-9:
                        return False
                else:
                    if abs(val) > 1e-9:
                        return False
    return True


def _is_half_integer(val, tol=1e-9):
    """Check if val is an integer or half-integer."""
    val2 = val * 2
    return abs(val2 - round(val2)) < tol


def _try_rank(n, m, p, target_rank):
    """Try to find a decomposition of the given rank."""
    # This is a placeholder for a search algorithm
    # For now, just return None to indicate failure
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a decomposition with rank less than schoolbook
    # For now, use schoolbook as fallback
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
