import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # T should equal the matrix multiplication tensor
    # T[(i,n*m)+(j,m*p)+k] ... actually let's think about the tensor structure
    
    # For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # The tensor has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # Actually, the multiplication tensor is: T[i,j,k] = 1 if j == j (always true for fixed j)
    # More precisely, the tensor M_{i,j,k} = delta_{j,j} which is just 1 for all valid i,j,k
    # Wait, that's not right. The matrix multiplication tensor has:
    # M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So the tensor is all ones of shape (n, m, p)
    # The decomposition says: M[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    # But wait, the U, V, W shapes are:
    # U: (n*m, rank) - indexed by (i,j) pair
    # V: (m*p, rank) - indexed by (j,k) pair  
    # W: (p*n, rank) - indexed by (k,i) pair
    
    # So the tensor entry for (i,j,k) is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    
    # Rank 35 decomposition using Strassen's algorithm on 5 blocks of 2 columns
    rank = 7 * num_blocks
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 35)
    # V: (m*p, rank) = (20, 35)
    # W: (p*n, rank) = (20, 35)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Indices in U: 0:a, 1:b, 2:c, 3:d
    # Indices in V (local): 0:e, 1:f, 2:g, 3:h
    # Indices in W (local): 0:C00, 1:C01, 2:C10, 3:C11
    
    # Define the 7 Strassen products
    # Each product is defined by:
    # u_coeffs: dict or list of (index, coeff) for A
    # v_coeffs: dict or list of (index, coeff) for B
    # w_coeffs: dict or list of (index, coeff) for C
    
    # P1 = (a + d)(e + h)
    # P2 = (c + d)e
    # P3 = a(f - h)
    # P4 = d(g - e)
    # P5 = (a + b)h
    # P6 = (c - a)(e + f)
    # P7 = (b - d)(g + h)
    
    strassen_products = [
        # (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),      # P1
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),            # P2
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),            # P3
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),            # P4
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),            # P5
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),            # P6
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),            # P7
    ]
    
    for b in range(num_blocks):
        base_rank = b * 7
        
        # V index offset for this block:
        # e = B[0, 2b] -> V[0*10 + 2b]
        # f = B[0, 2b+1] -> V[0*10 + 2b+1]
        # g = B[1, 2b] -> V[1*10 + 2b]
        # h = B[1, 2b+1] -> V[1*10 + 2b+1]
        
        # W index offset for this block:
        # C[0, 2b] -> W[2b*2 + 0]
        # C[0, 2b+1] -> W[(2b+1)*2 + 0]
        # C[1, 2b] -> W[2b*2 + 1]
        # C[1, 2b+1] -> W[(2b+1)*2 + 1]
        
        for k in range(7):
            r = base_rank + k
            u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff = strassen_products[k]
            
            # Fill U
            for idx, coeff in zip(u_idx, u_coeff):
                U[idx, r] = coeff
                
            # Fill V
            # Map local V indices (0:e, 1:f, 2:g, 3:h) to global
            # e -> 2b, f -> 2b+1, g -> 10+2b, h -> 10+2b+1
            for idx, coeff in zip(v_idx, v_coeff):
                if idx == 0:
                    V[2*b, r] = coeff
                elif idx == 1:
                    V[2*b+1, r] = coeff
                elif idx == 2:
                    V[10+2*b, r] = coeff
                elif idx == 3:
                    V[10+2*b+1, r] = coeff
                    
            # Fill W
            # Map local W indices (0:C00, 1:C01, 2:C10, 3:C11) to global
            # W[k*n+i] where k is row of C, i is column of C
            # C00 -> W[0*2+2b] = W[2b]
            # C01 -> W[0*2+(2b+1)] = W[2b+1]
            # C10 -> W[1*2+2b] = W[2+2b]
            # C11 -> W[1*2+(2b+1)] = W[3+2b]
            for idx, coeff in zip(w_idx, w_coeff):
                if idx == 0:
                    W[2*b, r] = coeff
                elif idx == 1:
                    W[2*b+1, r] = coeff
                elif idx == 2:
                    W[2+2*b, r] = coeff
                elif idx == 3:
                    W[3+2*b, r] = coeff
                    
    # Verify the decomposition
    if time.time() < deadline - 5:
        valid = _verify_decomposition(U, V, W, n, m, p)
        if not valid:
            # Fallback to schoolbook if Strassen is invalid
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school))
            V_s = np.zeros((m * p, rank_school))
            W_s = np.zeros((p * n, rank_school))
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i*m+j, r] = 1
                        V_s[j*p+k, r] = 1
                        W_s[k*n+i, r] = 1
                        r += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
