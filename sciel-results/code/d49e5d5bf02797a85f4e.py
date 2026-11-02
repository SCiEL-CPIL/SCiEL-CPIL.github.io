"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each rank term r: contribution is U[:,r] outer V[:,r] outer W[:,r]
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if (i,j,k) corresponds to a valid multiplication term, 0 otherwise.
    
    rank = U_arr.shape[1]
    # Build the full tensor
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U_arr[:, r].reshape(n, m)
        v = V_arr[:, r].reshape(m, p)
        w = W_arr[:, r].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    # The matrix multiplication tensor: T[i,j,k] = 1 if the (i,k)-th entry of A*B 
    # depends on A[i,j]*B[j,k], which is always the case.
    # Actually, the tensor for matrix mult is: T[i,j,k] = delta_{i,j,k} where the 
    # multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this indexing)
    # Wait, no. The tensor is defined such that C = A*B means:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # But actually the standard tensor decomposition is:
    # T = sum_r u_r ⊗ v_r ⊗ w_r where T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # And for matrix multiplication, T[i,j,k] = 1 for all i,j,k? No.
    # 
    # Let me reconsider. The multiplication tensor M_{n,m,p} is defined by:
    # M[i,j,k] = 1 if the (i,k) entry of the product depends on A[i,j] * B[j,k]
    # But that's always true. The actual tensor is:
    # For each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So M[i,j,k] = 1 for all i,j,k.
    
    # Hmm, but that would mean the tensor is all ones, which has rank 1.
    # That can't be right. Let me reconsider the tensor definition.
    
    # The matrix multiplication tensor is a 3-tensor T of size (n*p) x (n*m) x (m*p)
    # where T[(i,k), (i,j), (j,k)] = 1.
    # In the decomposition, U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # and the tensor is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a indexes A entries (i*m+j), b indexes B entries (j*p+k), c indexes C entries (k*n+i)
    # And T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some valid indices, 0 otherwise.
    
    # So for our case: T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise.
    
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected[a, b, c] = 1
    
    # Compute from decomposition
    computed = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        computed += np.einsum('a,b,c->abc', u, v, w)
    
    max_err = np.max(np.abs(computed - expected))
    return max_err < 1e-9

def construct():
    n, m, p = DIMENSIONS
    # Start with the known rank-28 Strassen decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(4):
        base_col = b * 7
        v00 = 2*b
        v01 = 2*b+1
        v10 = 8+2*b
        v11 = 8+2*b+1
        w00 = 4*b
        w10 = 4*b+1
        w01 = 4*b+2
        w11 = 4*b+3
        
        U[0][base_col] = 1
        U[3][base_col] = 1
        V[v00][base_col] = 1
        V[v11][base_col] = 1
        W[w00][base_col] = 1
        W[w11][base_col] = 1
        
        U[2][base_col+1] = 1
        U[3][base_col+1] = 1
        V[v00][base_col+1] = 1
        W[w10][base_col+1] = 1
        W[w11][base_col+1] = -1
        
        U[0][base_col+2] = 1
        V[v01][base_col+2] = 1
        V[v11][base_col+2] = -1
        W[w01][base_col+2] = 1
        W[w11][base_col+2] = 1
        
        U[3][base_col+3] = 1
        V[v10][base_col+3] = 1
        V[v00][base_col+3] = -1
        W[w00][base_col+3] = 1
        W[w10][base_col+3] = 1
        
        U[0][base_col+4] = 1
        U[1][base_col+4] = 1
        V[v11][base_col+4] = 1
        W[w00][base_col+4] = -1
        W[w01][base_col+4] = 1
        
        U[2][base_col+5] = 1
        U[0][base_col+5] = -1
        V[v00][base_col+5] = 1
        V[v01][base_col+5] = 1
        W[w11][base_col+5] = 1
        
        U[1][base_col+6] = 1
        U[3][base_col+6] = -1
        V[v10][base_col+6] = 1
        V[v11][base_col+6] = 1
        W[w00][base_col+6] = 1
    
    # Verify the base decomposition
    assert _verify_decomposition(U, V, W, n, m, p), "Base decomposition failed verification"
    
    # Try to find a lower-rank decomposition by checking if we can
    # combine or eliminate rank terms
    # Strategy: check if any rank term is a linear combination of others
    # in the tensor space
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor is T = sum_r U_r ⊗ V_r ⊗ W_r
    # We want to find the minimum number of terms needed.
    # This is equivalent to finding the rank of the tensor.
    
    # For a small tensor, we can try to find a lower-rank decomposition
    # by solving a system of polynomial equations, but that's hard.
    
    # Instead, let's try a different approach: use a different block structure.
    # Try splitting the 8 columns into two groups of 4, and use a 2x2 * 2x4 
    # multiplication with a lower-rank algorithm.
    
    # For 2x2 * 2x4: The tensor has size (2*2)*(2*4)*(4*2) = 4*8*8 = 256 entries
    # Schoolbook rank: 2*2*4 = 16
    # Can we do better?
    
    # Let's try: split the 4 columns into two pairs, use Strassen on each pair.
    # That gives 2*7 = 14 for the 2x4 block. Then 2x2 * 2x8 = 2 * 14 = 28. Same.
    
    # Alternative: use a 2x2 * 2x4 with a custom algorithm.
    # Let A = [[a,b],[c,d]], B = [[e,f,g,h],[i,j,k,l]]
    # C = [[ae+bi, af+bj, ag+bk, ah+bl],
    #      [ce+di, cf+dj, cg+dk, ch+dl]]
    
    # Can we compute this with fewer than 16 multiplications?
    # Try: 
    # P1 = (a+d)(e+i)
    # P2 = (c+d)e
    # P3 = a(f-i)
    # P4 = d(j-e)
    # P5 = (a+b)l
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(i+j)
    # P8 = (a+d)(g+k)
    # P9 = (c+d)g
    # P10 = a(h-k)
    # P11 = d(l-g)
    # P12 = (a+b)k  -- wait, this overlaps with P5
    # P13 = (c-a)(g+h)
    # P14 = (b-d)(k+l)
    
    # This is getting complicated. Let me just try to verify if rank 24 is possible
    # by attempting a direct search.
    
    # For now, return the verified rank-28 solution.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
