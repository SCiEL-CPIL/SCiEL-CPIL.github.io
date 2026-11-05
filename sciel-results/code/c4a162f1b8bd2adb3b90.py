import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 36.
    # We attempt a rank 34 decomposition.
    
    # If we can't verify quickly, we might return schoolbook.
    # But 34 is a known improvement.
    
    # Let's build the rank 34 decomposition.
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    def add_term(u_vals, v_vals, w_vals):
        # u_vals: dict of (i,j) -> val
        # v_vals: dict of (j,k) -> val
        # w_vals: dict of (i,k) -> val
        u_row = [0] * (n * m)
        v_row = [0] * (m * p)
        w_row = [0] * (p * n)
        
        for (i, j), val in u_vals.items():
            u_row[i * m + j] = val
        for (j, k), val in v_vals.items():
            v_row[j * p + k] = val
        for (i, k), val in w_vals.items():
            w_row[k * n + i] = val
            
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)

    # Block 1: A11 (2x2) * B11 (2x2) -> C11 (2x2)
    # Strassen 7 terms
    # A11 indices: (0,0), (0,1), (1,0), (1,1)
    # B11 indices: (0,0), (0,1), (1,0), (1,1)
    # C11 indices: (0,0), (0,1), (1,0), (1,1)
    
    # M1
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # M2
    add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1})
    # M3
    add_term({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1})
    # M4
    add_term({(1,1):1}, {(0,0):1, (1,0):-1}, {(0,0):1, (1,0):1})
    # M5
    add_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})
    # M6
    add_term({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1})
    # M7
    add_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):-1})

    # Block 2: A12 (2x1) * B21 (1x2) -> C11 (2x2) part
    # A12 indices: (0,2), (1,2)
    # B21 indices: (2,0), (2,1)
    # C11 indices: (0,0), (0,1), (1,0), (1,1)
    # Schoolbook 4 terms
    add_term({(0,2):1}, {(2,0):1}, {(0,0):1})
    add_term({(0,2):1}, {(2,1):1}, {(0,1):1})
    add_term({(1,2):1}, {(2,0):1}, {(1,0):1})
    add_term({(1,2):1}, {(2,1):1}, {(1,1):1})

    # Block 3: A11 (2x2) * B12 (2x2) -> C12 (2x2)
    # A11 indices: (0,0), (0,1), (1,0), (1,1)
    # B12 indices: (0,2), (0,3), (1,2), (1,3)
    # C12 indices: (0,2), (0,3), (1,2), (1,3)
    # Strassen 7 terms
    add_term({(0,0):1, (1,1):1}, {(0,2):1, (1,3):1}, {(0,2):1, (1,3):1})
    add_term({(1,0):1, (1,1):1}, {(0,2):1}, {(1,2):1, (1,3):-1})
    add_term({(0,0):1}, {(1,2):1, (1,3):-1}, {(0,3):1, (1,3):1})
    add_term({(1,1):1}, {(0,2):1, (1,2):-1}, {(0,2):1, (1,2):1})
    add_term({(0,0):1, (0,1):1}, {(1,3):1}, {(0,2):-1, (0,3):1})
    add_term({(1,0):1, (0,0):-1}, {(0,2):1, (1,2):1}, {(1,3):1})
    add_term({(0,1):1, (1,1):-1}, {(1,2):1, (1,3):1}, {(0,2):1, (0,3):-1})

    # Block 4: A12 (2x1) * B22 (1x2) -> C12 (2x2) part
    # A12 indices: (0,2), (1,2)
    # B22 indices: (2,2), (2,3)
    # C12 indices: (0,2), (0,3), (1,2), (1,3)
    # Schoolbook 4 terms
    add_term({(0,2):1}, {(2,2):1}, {(0,2):1})
    add_term({(0,2):1}, {(2,3):1}, {(0,3):1})
    add_term({(1,2):1}, {(2,2):1}, {(1,2):1})
    add_term({(1,2):1}, {(2,3):1}, {(1,3):1})

    # Block 5: A21 (1x2) * B11 (2x2) -> C21 (1x2)
    # A21 indices: (2,0), (2,1)
    # B11 indices: (0,0), (0,1), (1,0), (1,1)
    # C21 indices: (2,0), (2,1)
    # Schoolbook 4 terms
    add_term({(2,0):1}, {(0,0):1}, {(2,0):1})
    add_term({(2,0):1}, {(1,0):1}, {(2,0):1})
    add_term({(2,1):1}, {(0,1):1}, {(2,1):1})
    add_term({(2,1):1}, {(1,1):1}, {(2,1):1})

    # Block 6: A22 (1x1) * B21 (1x2) -> C21 (1x2) part
    # A22 indices: (2,2)
    # B21 indices: (2,0), (2,1)
    # C21 indices: (2,0), (2,1)
    # Schoolbook 2 terms
    add_term({(2,2):1}, {(2,0):1}, {(2,0):1})
    add_term({(2,2):1}, {(2,1):1}, {(2,1):1})

    # Block 7: A21 (1x2) * B12 (2x2) -> C22 (1x2)
    # A21 indices: (2,0), (2,1)
    # B12 indices: (0,2), (0,3), (1,2), (1,3)
    # C22 indices: (2,2), (2,3)
    # Schoolbook 4 terms
    add_term({(2,0):1}, {(0,2):1}, {(2,2):1})
    add_term({(2,0):1}, {(1,2):1}, {(2,2):1})
    add_term({(2,1):1}, {(0,3):1}, {(2,3):1})
    add_term({(2,1):1}, {(1,3):1}, {(2,3):1})

    # Block 8: A22 (1x1) * B22 (1x2) -> C22 (1x2) part
    # A22 indices: (2,2)
    # B22 indices: (2,2), (2,3)
    # C22 indices: (2,2), (2,3)
    # Schoolbook 2 terms
    add_term({(2,2):1}, {(2,2):1}, {(2,2):1})
    add_term({(2,2):1}, {(2,3):1}, {(2,3):1})

    rank = len(U)
    
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check rank
    # The decomposition is exact if sum_r U[:,r] V[:,r] W[:,r] == T
    # T[i*m+j, j*p+k, k*n+i] = delta? No, T is the multiplication tensor.
    # T_{ijk} = 1 if the indices match a valid multiplication term.
    # Specifically, the tensor is defined such that C = AB.
    # C_ik = sum_j A_ij B_jk.
    # So T[i, j, k] is 1 if we are computing C_ik from A_ij and B_jk.
    # In the factorization U, V, W, we have:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) is a valid term, 0 otherwise?
    # No, the tensor T has entries T_{(i,j),(j,k),(k,i)}?
    # The task says: "exactly decompose the matrix multiplication tensor".
    # The tensor T is of size (n*m) x (m*p) x (p*n).
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
    # Note that b's first index must equal a's second index.
    # And c's second index must equal a's first index.
    # And c's first index must equal b's second index.
    
    # Let's verify with a random matrix.
    A_rand = np.random.rand(n, m)
    B_rand = np.random.rand(m, p)
    C_rand = A_rand @ B_rand
    
    # Compute C via decomposition
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u = U_np[:, r].reshape(n, m)
        v = V_np[:, r].reshape(m, p)
        w = W_np[:, r].reshape(p, n)
        # The term is u * v * w? No.
        # The rank-1 term is u \otimes v \otimes w.
        # The contribution to C is sum_{j} u[i,j] v[j,k] w[k,i]?
        # No, the decomposition is sum_r U[:,r] V[:,r] W[:,r].
        # This is a tensor.
        # The multiplication is C_ik = sum_j A_ij B_jk.
        # The tensor T is such that T_{(i,j),(j,k),(k,i)} = 1.
        # So sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be 1 if the indices are consistent, 0 otherwise?
        # Actually, the condition is that for any A, B:
        # C = AB
        # C_ik = sum_j A_ij B_jk
        # Also C_ik = sum_r (sum_j A_ij U[i*m+j, r]) (sum_k B_jk V[j*p+k, r]) ... no.
        
        # The standard form is:
        # C_ik = sum_r (sum_j A_ij U[i*m+j, r]) (sum_j B_jk V[j*p+k, r]) W[k*n+i, r]
        # This is not correct.
        
        # The correct interpretation:
        # The tensor T is decomposed as sum_r u_r \otimes v_r \otimes w_r.
        # T_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r].
        # For matrix multiplication, T_{(i,j),(j,k),(k,i)} = 1, and 0 otherwise.
        # So we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if i,j,k are valid, 0 otherwise.
        
        pass
        
    # Simple verification:
    # Construct the tensor from U,V,W
    # and check if it matches the multiplication tensor.
    
    # Multiplication tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
                
    # Decomposed tensor
    T_decomp = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    if not np.allclose(T, T_decomp):
        # Fallback to schoolbook
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

    return {"U": U, "V": V, "W": W}
