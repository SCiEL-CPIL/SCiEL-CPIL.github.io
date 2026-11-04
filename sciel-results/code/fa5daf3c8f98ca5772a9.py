import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if i,j,k correspond to the multiplication term, else 0
    # Actually, the tensor T for matrix multiplication is defined by T[i, j, k] = 1
    # where the term is A[i, j] * B[j, k] * C[k, i] ... wait.
    # The tensor decomposition is:
    # (A B)[i, k] = sum_j A[i, j] B[j, k]
    # This can be written as sum_c U[i, c] V[j, c] W[k, c] ?? No.
    # The standard form is:
    # sum_c (U_c \otimes V_c \otimes W_c)
    # where (U_c \otimes V_c \otimes W_c)[i, j, k] = U_c[i] V_c[j] W_c[k]
    # We want sum_c U_c[i] V_c[j] W_c[k] = delta_{j, j'} ... no.
    # The tensor T is such that T[i, j, k] is the coefficient of A[i, j] B[j, k] in the product?
    # No, the matrix multiplication tensor M_{n,m,p} is defined by:
    # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i \otimes e_j \otimes e_k
    # where e_i are basis vectors.
    # Wait, the product C = AB is C[i, k] = sum_j A[i, j] B[j, k].
    # This is a bilinear map.
    # The tensor T is in V* \otimes W* \otimes Z where dim V=n*m, dim W=m*p, dim Z=p*n.
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
    # Specifically, a = i*m + j, b = j*p + k, c = k*n + i.
    # There is exactly one non-zero entry for each (i,j,k) triple?
    # No, for a fixed (i,j,k), a is fixed, b is fixed, c is fixed.
    # So T[a, b, c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i.
    # Note that j must be the same in a and b.
    # k must be the same in b and c.
    # i must be the same in c and a.
    # So for each (i,j,k), there is one 1 in the tensor.
    # Total non-zeros = n*m*p = 60.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1

    # Try to find a lower rank decomposition
    # We will try ranks from 59 down to 54
    # Use random restarts and gradient descent
    
    def tensor_norm(U, V, W):
        # Compute sum_c U[:, c] \otimes V[:, c] \otimes W[:, c]
        # and subtract T
        # Efficiently:
        # T_approx[a, b, c] = sum_u U[a, u] V[b, u] W[c, u]
        # This is a 3-tensor contraction.
        # Can be done as:
        # Temp = U @ V.T  (n*m x m*p) ... no, U is (n*m, rank), V is (m*p, rank)
        # U @ V.T is (n*m, m*p) with entries sum_u U[a, u] V[b, u]
        # Then we need to contract with W.
        # Result[a, b, c] = sum_u (U[a, u] V[b, u]) W[c, u]
        # Let M_ab = sum_u U[a, u] V[b, u]
        # Then Result[a, b, c] = sum_u M_ab ... no, M depends on u.
        # We can compute:
        # X = U @ V.T  # (n*m, m*p)
        # This is sum_u U[a, u] V[b, u].
        # But we need sum_u U[a, u] V[b, u] W[c, u].
        # This is not a simple matrix product.
        # We can iterate over u:
        # Res = np.zeros((n*m, m*p, p*n))
        # for u in range(rank):
        #     Res += np.outer(np.outer(U[:, u], V[:, u]), W[:, u])
        # This is O(rank * n*m * m*p * p*n).
        # 60 * 60 * 20 * 15 = 108,000 operations. Very fast.
        pass

    # Let's implement the norm calculation efficiently
    def compute_error(U, V, W):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Error = || T - sum_u U[:,u] \otimes V[:,u] \otimes W[:,u] ||_F
        # We can compute the approximation A
        A = np.zeros((n*m, m*p, p*n))
        for u in range(U.shape[1]):
            # Outer product
            # u_vec = U[:, u]
            # v_vec = V[:, u]
            # w_vec = W[:, u]
            # A += np.einsum('i,j,k->ijk', U[:, u], V[:, u], W[:, u])
            A += np.multiply.outer(np.multiply.outer(U[:, u], V[:, u]), W[:, u])
        diff = T - A
        return np.linalg.norm(diff)

    # Gradient descent step
    def grad_step(U, V, W, lr):
        # Compute gradients
        # E = T - A
        # dU[i, u] = - sum_{b,c} E[i, b, c] V[b, u] W[c, u]
        # dV[j, u] = - sum_{a,c} E[a, j, c] U[a, u] W[c, u]
        # dW[k, u] = - sum_{a,b} E[a, b, k] U[a, u] V[b, u]
        
        A = np.zeros((n*m, m*p, p*n))
        for u in range(U.shape[1]):
            A += np.multiply.outer(np.multiply.outer(U[:, u], V[:, u]), W[:, u])
        E = T - A
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for u in range(U.shape[1]):
            # dU[:, u] = - E @ (V[:, u] \otimes W[:, u])?
            # E[i, b, c] V[b, u] W[c, u]
            # Let M_u[b, c] = V[b, u] W[c, u]
            # dU[i, u] = - sum_{b,c} E[i, b, c] M_u[b, c]
            M_u = np.outer(V[:, u], W[:, u])
            # E is (n*m, m*p, p*n)
            # We need to contract E with M_u over b, c.
            # result[i] = sum_{b,c} E[i, b, c] M_u[b, c]
            # This is like a matrix multiplication if we reshape E.
            # E reshaped to (n*m, m*p*p*n)
            # M_u reshaped to (m*p*p*n)
            # dU[:, u] = - (E.reshape(n*m, -1) @ M_u.flatten())
            dU[:, u] = - (E.reshape(n*m, -1) @ M_u.flatten())
            
            # dV[j, u] = - sum_{a,c} E[a, j, c] U[a, u] W[c, u]
            # N_u[a, c] = U[a, u] W[c, u]
            N_u = np.outer(U[:, u], W[:, u])
            # E[a, j, c]
            # We need to contract over a, c.
            # E reshaped to (m*p, n*m*p*n)? No.
            # E indices: a (n*m), j (m*p), c (p*n)
            # We want sum_{a,c} E[a, j, c] N_u[a, c]
            # Move j to front: E_reshaped[j, a, c]
            E_reshaped = E.transpose(1, 0, 2) # (m*p, n*m, p*n)
            dV[:, u] = - (E_reshaped.reshape(m*p, -1) @ N_u.flatten())
            
            # dW[k, u] = - sum_{a,b} E[a, b, k] U[a, u] V[b, u]
            P_u = np.outer(U[:, u], V[:, u])
            # E[a, b, k]
            # Move k to front: E_reshaped2[k, a, b]
            E_reshaped2 = E.transpose(2, 0, 1) # (p*n, n*m, m*p)
            dW[:, u] = - (E_reshaped2.reshape(p*n, -1) @ P_u.flatten())
            
        U = U - lr * dU
        V = V - lr * dV
        W = W - lr * dW
        return U, V, W

    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try ranks 59 down to 54
    for rank in range(rank_school - 1, 53, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for _ in range(20):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Scale to match magnitude
            # T has 60 ones.
            # Approximation should have similar magnitude.
            # Random init might be too large.
            U *= 0.1
            V *= 0.1
            W *= 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                U, V, W = grad_step(U, V, W, lr)
                # Decay learning rate
                if step % 100 == 0:
                    lr *= 0.5
            
            # Check error
            err = compute_error(U, V, W)
            if err < 1e-5:
                # Try rounding to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                err_int = compute_error(U_int, V_int, W_int)
                if err_int < 1e-9:
                    best_rank = rank
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    # If we found a low rank, we can break early if we want, 
                    # but we want the minimum.
                    # Since we iterate downwards, the first found is the best.
                    # But we might find 59, then 58, etc.
                    # Let's continue to see if we can go lower.
                    # However, if we find 54, we can stop.
                    if rank == 54:
                        break
            # If we found a valid solution for this rank, we can break the init loop
            # and try lower rank.
            if best_rank == rank:
                break

    # Convert to lists of lists as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
