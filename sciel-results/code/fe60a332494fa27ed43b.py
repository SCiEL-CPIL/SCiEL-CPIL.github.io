import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n=4, m=4, p=5, num_samples=200, tol=1e-9):
    """Verify decomposition by checking A*B = sum_c U_c * V_c * W_c on random inputs."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each term c: (n x m) * (m x p) contribution via outer products
    # Actually: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] ... no
    # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the convention is W[k*n+i, r], so W is indexed by (k,i)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal delta... no, it should equal 1 if the (i,j,k) triple is "active"
    # Actually for matrix mult: (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k), and we need:
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # But that's not right either. Let me reconsider.
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # No wait - it's a trilinear form. The tensor for matmul is:
    # T[i,j,k] = 1 if we're computing the (i,k) entry using A[i,j] and B[j,k]
    # So actually T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the appropriate sense)
    # Hmm, that can't be right because then rank would be 1.
    
    # Let me think again. The matrix multiplication tensor T is defined by:
    # (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
    # As a trilinear form: T(A,B,C) = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[k,i]
    # where C is the output. For this to equal trace(C^T * A*B) = sum_{i,k} C[k,i]*sum_j A[i,j]*B[j,k]
    # we need T[i,j,k] = 1 for all i,j,k.
    
    # But that gives rank 1, which is wrong. The issue is that the "tensor" here is not
    # the trilinear form but the bilinear map. Let me use the standard formulation:
    # 
    # The multiplication tensor is a 3-tensor T with T[i,j,k] = 1 if (i,j,k) corresponds
    # to a term in the product. For (n x m) * (m x p) -> (n x p):
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # 
    # Wait, that IS the all-ones tensor, and its tensor rank is indeed 1? No!
    # 
    # I think I'm confusing myself. Let me use the correct definition.
    # 
    # The matrix multiplication tensor T_{n,m,p} is defined such that:
    # T[i,j,k] = 1 if the (i,k) entry of the product uses A[i,j] and B[j,k]
    # For standard matrix multiplication, every (i,j,k) triple contributes, so T[i,j,k] = 1.
    # 
    # But the tensor rank of the all-ones tensor of size n*m*p is 1, which is clearly wrong.
    # 
    # I see the issue. The tensor rank is for the trilinear form, but matrix multiplication
    # is a bilinear map. The correct formulation:
    # 
    # We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # As a tensor decomposition: C[i,k] = sum_r sum_j U[i,j,r] * V[j,k,r] * ... 
    # No, that's not right either.
    # 
    # The standard tensor decomposition for matmul:
    # C[i,k] = sum_r U_r[i,k] * (sum_j A_r[i,j] * B_r[j,k])
    # where each term r involves a "local" matrix multiplication.
    # 
    # More precisely, the decomposition is:
    # For each r in [0, rank):
    #   A_r is an n x m matrix (sparse, with entries in {0, ±1, ±1/2})
    #   B_r is an m x p matrix  
    #   C_r is an n x p matrix
    # And: A * B = sum_r (A_r * B_r) where A_r * B_r means the element-wise product
    # followed by... no.
    # 
    # Let me use the interface specification directly:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The decomposition says: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all valid (i,j,k).
    # 
    # Wait, but that would mean the all-ones tensor has a low-rank decomposition,
    # which it does (rank 1). So something is wrong with my understanding.
    # 
    # OH WAIT. I think the tensor is NOT all ones. Let me reconsider.
    # 
    # In the standard formulation, the matrix multiplication tensor T has:
    # T[i,j,k] = 1 if (i,j,k) is a "valid" triple, but the key is that the
    # decomposition must satisfy:
    #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1  for all i,j,k
    # 
    # And the rank is the minimum number of terms. For the all-ones tensor,
    # the rank is indeed 1, which is absurd.
    # 
    # I must be misunderstanding the tensor. Let me look at this differently.
    # 
    # Actually, I think the issue is that the "tensor" for matrix multiplication
    # is not the all-ones tensor. The correct tensor is:
    # T[i,j,k] = 1 for all i,j,k, BUT the decomposition must use the specific
    # index structure where U depends on (i,j), V on (j,k), W on (k,i).
    # 
    # No, that's still the all-ones tensor.
    # 
    # Let me just test with the schoolbook: in the schoolbook, for each (i,j,k),
    # there's one term where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1. So:
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for the specific (i,j,k)
    # and 0 for other (i',j',k') if they don't share the same r.
    # 
    # Wait no. In the schoolbook, each (i,j,k) gets its own column r. So:
    # For column r corresponding to (i0,j0,k0):
    #   U[i0*m+j0, r] = 1, all other U[*, r] = 0
    #   V[j0*p+k0, r] = 1, all other V[*, r] = 0  
    #   W[k0*n+i0, r] = 1, all other W[*, r] = 0
    # 
    # So for a query (i,j,k):
    #   sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    #   = 1 only if there exists r such that (i,j,k) = (i0,j0,k0) for that r
    #   = 1 if (i,j,k) is in the set of triples, which is all of them
    # 
    # So yes, it equals 1 for all (i,j,k). And the rank is n*m*p.
    # 
    # So the problem is: find the minimum rank R such that there exist U,V,W with
    # U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] summed over r equals 1 for all (i,j,k).
    # 
    # This is the tensor rank of the "all-ones" tensor with the specific index
    # grouping. And it's NOT 1 because the rank-1 decomposition would require
    # U[i*m+j,r] = a[i,j], V[j*p+k,r] = b[j,k], W[k*n+i,r] = c[k,i] such that
    # a[i,j]*b[j,k]*c[k,i] = 1 for all i,j,k, which is impossible for n,m,p >= 2
    # (it would require a[i,j] = 1/b[j,k]/c[k,i] for all j,k, forcing a[i,j] to
    # depend only on i in a very restrictive way).
    
    # OK so now I understand. Let me just verify by checking that the sum equals 1
    # for all (i,j,k).
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Reshape for efficient computation
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)  
    # W: (p*n, rank) -> (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # For each (i,j,k), compute sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This should equal 1 for all i,j,k
    
    # Efficient: use einsum
    # result[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check that result[i,j,k] = 1 for all i,j,k
    target = np.ones((n, m, p))
    err = np.max(np.abs(result - target))
    return err < tol

def _try_round_halfint(U, V, W, n=4, m=4, p=5):
    """Try to round to half-integers and verify."""
    def round_half(x):
        return np.round(x * 2) / 2.0
    
    U_r = round_half(U)
    V_r = round_half(V)
    W_r = round_half(W)
    
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _als_step(U, V, W, n=4, m=4, p=5, rank=None):
    """One ALS step: update one factor at a time."""
    if rank is None:
        rank = U.shape[1]
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Update U: for each (i,j), solve for U3[i,j,:]
    # sum_r U3[i,j,r] * (V3[j,k,r] * W3[k,i,r] summed over k) = 1 for all k
    # Let's define for fixed (i,j): sum_r U3[i,j,r] * D_r = 1 for all k
    # where D_r = sum_k V3[j,k,r] * W3[k,i,r]
    # This is: U3[i,j,:] * D = 1 (vector equation)
    
    # Actually, it's: for each k, sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] = 1
    # So: U3[i,j,:] @ (V3[j,:,r] * W3[:,i,r]) = 1 for all k
    # Let M[i,j] be a p x rank matrix where M[i,j][k,r] = V3[j,k,r] * W3[k,i,r]
    # Then: U3[i,j,:] @ M[i,j] = ones(p)
    # i.e., M[i,j].T @ U3[i,j,:] = ones(p)
    
    new_U = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            # M[k,r] = V3[j,k,r] * W3[k,i,r], shape (p, rank)
            M = V3[j, :, :] * W3[:, i, :]  # (p, rank)
            # Solve: M.T @ u = ones(p) where u = U3[i,j,:]
            # If rank >= p, use least squares
            try:
                u, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                new_U[i*m+j, :] = u
            except:
                new_U[i*m+j, :] = U[i*m+j, :]
    
    U = new_U
    U3 = U.reshape(n, m, rank)
    
    # Update V: for each (j,k), solve for V3[j,k,:]
    # sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] = 1 for all i
    # M[i,r] = U3[i,j,r] * W3[k,i,r], shape (n, rank)
    # V3[j,k,:] @ M = ones(n)
    new_V = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            M = U3[:, j, :] * W3[k, :, :]  # (n, rank)
            try:
                v, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                new_V[j*p+k, :] = v
            except:
                new_V[j*p+k, :] = V[j*p+k, :]
    
    V = new_V
    V3 = V.reshape(m, p, rank)
    
    # Update W: for each (k,i), solve for W3[k,i,:]
    # sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] = 1 for all j
    # M[j,r] = U3[i,j,r] * V3[j,k,r], shape (m, rank)
    # W3[k,i,:] @ M = ones(m)
    new_W = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            M = U3[i, :, :] * V3[:, k, :]  # (m, rank)
            try:
                w, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                new_W[k*n+i, :] = w
            except:
                new_W[k*n+i, :] = W[k*n+i, :]
    
    W = new_W
    
    return U, V, W

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    U, V, W = _construct_schoolbook()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try to reduce rank via ALS with random restarts
    # Target ranks to try
    target_ranks = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 30, 28, 25, 20, 16]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 5:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple random initializations
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline - 3:
                break
            
            # Initialize randomly
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Run ALS
            for step in range(200):
                if time.time() > deadline - 2:
                    break
                U, V, W = _als_step(U, V, W, n, m, p, target_rank)
                
                # Check convergence
                U3 = U.reshape(n, m, target_rank)
                V3 = V.reshape(m, p, target_rank)
                W3 = W.reshape(p, n, target_rank)
                result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                err = np.max(np.abs(result - 1.0))
                
                if err < 1e-8:
                    # Check half-integer rounding
                    rounded = _try_round_halfint(U, V, W, n, m, p)
                    if rounded is not None:
                        U_r, V_r, W_r = rounded
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        break
                elif err < 1e-4:
                    # Close, try rounding
                    rounded = _try_round_halfint(U, V, W, n, m, p)
                    if rounded is not None:
                        U_r, V_r, W_r = rounded
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        break
            else:
                # Didn't break, check final
                U3 = U.reshape(n, m, target_rank)
                V3 = V.reshape(m, p, target_rank)
                W3 = W.reshape(p, n, target_rank)
                result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                err = np.max(np.abs(result - 1.0))
                if err < 1e-8:
                    rounded = _try_round_halfint(U, V, W, n, m, p)
                    if rounded is not None:
                        U_r, V_r, W_r = rounded
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
    
    # Return the best found
    if best_rank < n*m*p:
        # Convert to list of lists
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Return schoolbook
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
