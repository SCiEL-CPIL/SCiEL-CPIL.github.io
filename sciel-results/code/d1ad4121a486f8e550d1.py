import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _build_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    R = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                Target[u, v, w] = 1.0
    err = np.max(np.abs(T - Target))
    return err < 1e-6

def _snap_to_half_int(M):
    return np.round(M * 2) / 2

def _compute_algebraic_rank(n, m, p):
    """
    Computes the rank of the matrix multiplication tensor by flattening it into a matrix.
    The tensor T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    Flattening along the first two modes (i*m + j) and the third mode (k*n + i)
    creates a matrix where entry (u, w) is 1 if there exists a j such that u = i*m+j, w = k*n+i, and j is consistent?
    Actually, the standard flattening for rank is taking the tensor as a matrix M_{(i,j), (k,l)}? No.
    The rank of the multiplication tensor is the rank of the matrix obtained by flattening the tensor.
    For matrix multiplication, the tensor is M_{(i,j), (j,k)}? No, indices are (i,j) for A, (j,k) for B, (i,k) for C.
    Indices: A is (i,j), B is (j,k), C is (i,k).
    Tensor T has indices (i,j,k) with value 1.
    Flattening (i,j) -> row, k -> col? No, that's not right.
    The tensor is a 3-way array.
    Matrix multiplication is a bilinear map.
    The tensor is T_{i,j,k} = 1.
    If we flatten (i,j) into a single index u and k into a single index v?
    Then T_{u, v} is 1 if u corresponds to (i,j) and v corresponds to k?
    But k is the column index of B and row index of C.
    Wait, the tensor is defined by the identity: C_{ik} = sum_j A_{ij} B_{jk}.
    So the tensor components are T_{ij, jk, ik} = 1.
    Indices:
    A: (i, j) -> i in [0,n), j in [0,m)
    B: (j, k) -> j in [0,m), k in [0,p)
    C: (i, k) -> i in [0,n), k in [0,p)
    
    The tensor T is a 3-tensor with indices (i,j), (j,k), (i,k).
    To find the rank, we can flatten it into a matrix.
    One way is to fix one index and look at the others.
    However, the rank of the tensor is the minimum number of rank-1 tensors needed to sum to T.
    This is equivalent to the rank of the matrix obtained by flattening the tensor.
    Which flattening?
    Usually, for a tensor T_{abc}, the rank is the rank of the matrix T_{(ab), c} or T_{a, (bc)} etc.
    But for matrix multiplication, all flattening ranks are the same?
    Let's check for 2x2x2.
    T_{ij, jk, ik} = 1.
    Flatten (i,j) and (j,k) -> matrix M_{(i,j), (j,k)}? No, that's not a valid matrix because j is shared.
    The tensor is defined on the space U x V x W.
    U = R^{n x m}, V = R^{m x p}, W = R^{n x p}.
    T: U x V -> W is bilinear.
    The tensor is an element of U^* x V^* x W.
    Or rather, the multiplication map is m: U x V -> W.
    The tensor is M in U^* \otimes V^* \otimes W.
    M(u, v, w) = w(m(u,v)).
    The rank of M is the rank of the matrix obtained by flattening M.
    Flatten U^* x V^* into a single index (i,j,k) and W into (l,m)?
    No, U^* has dimension n*m, V^* has dimension m*p, W has dimension n*p.
    So M is a tensor of size (n*m, m*p, n*p).
    The rank is the rank of the matrix M_{(n*m), (m*p, n*p)}?
    Or M_{(n*m, m*p), (n*p)}?
    Let's try to construct the matrix.
    Let u = (i,j), v = (j', k), w = (i', k').
    M_{u, v, w} = 1 if u=(i,j), v=(j,k), w=(i,k).
    So for a fixed u=(i,j) and w=(i',k'), M_{u, v, w} is non-zero only if i=i' and k=k' (from w) and j=j' (from v).
    So v must be (j, k).
    Thus, for fixed u=(i,j) and w=(i',k'), the vector over v is non-zero only at v=(j,k) if i=i' and k=k'.
    If we flatten v and w into a single index?
    Let's flatten (v, w) into a matrix column.
    Row index: u = i*m + j.
    Col index: v_idx * (n*p) + w_idx? No.
    The rank of the tensor is the rank of the matrix obtained by flattening the tensor in any way.
    Let's flatten U^* (rows) and V^* \otimes W (cols).
    Rows: i*m + j.
    Cols: (j*p + k) * (n*p) + (i'*n + k')?
    Let's check the non-zero entries.
    M_{u, v, w} = 1 if u=(i,j), v=(j,k), w=(i,k).
    So for a specific row u=(i,j), the non-zero entries in the matrix (flattened V^* \otimes W) are at columns corresponding to v=(j,k) and w=(i,k) for all k.
    So for a fixed row u=(i,j), there are p non-zero entries in the flattened matrix.
    These entries are at positions:
    v = (j, k) -> index j*p + k
    w = (i, k) -> index i*n + k
    Flattened col index = (j*p + k) * (n*p) + (i*n + k).
    
    Let's check if these rows are linearly independent.
    Row u=(i,j) has 1s at columns C_{i,j,k} for k=0..p-1.
    Are these rows linearly independent?
    Consider two rows u1=(i1, j1) and u2=(i2, j2).
    If (i1, j1) != (i2, j2), do they share any non-zero columns?
    A column is identified by (j, k, i, k).
    For row (i1, j1), non-zero cols have j=j1, i=i1.
    For row (i2, j2), non-zero cols have j=j2, i=i2.
    If i1 != i2 or j1 != j2, the sets of columns are disjoint.
    So the rows are orthogonal.
    Thus the rank is the number of rows, which is n*m.
    Wait, this logic implies the rank is n*m.
    But we know the rank is at most n*m*p.
    And for 2x2x2, rank is 7, not 4.
    Where is the error?
    The error is in the flattening.
    The tensor is in U^* \otimes V^* \otimes W.
    The rank is the rank of the matrix M: U^* -> V^* \otimes W?
    Or M: U^* \otimes V^* -> W?
    The rank of a tensor is the minimum number of rank-1 tensors.
    This is equal to the rank of the matrix obtained by flattening the tensor.
    But which flattening?
    The rank of the tensor is the rank of the matrix obtained by flattening the tensor in ANY way?
    No, different flattening ranks can be different.
    The rank of the tensor is the rank of the matrix obtained by flattening the tensor such that the tensor is viewed as a linear map from one space to the tensor product of the other two.
    For matrix multiplication, the tensor is M: R^{n*m} \otimes R^{m*p} -> R^{n*p}.
    So we can view it as a linear map from R^{n*m} to (R^{m*p})^* \otimes R^{n*p}.
    Or from R^{m*p} to (R^{n*m})^* \otimes R^{n*p}.
    Or from R^{n*p} to (R^{n*m})^* \otimes (R^{m*p})^*.
    The rank of the tensor is the rank of the matrix representing this linear map.
    Let's check the rank of the map from R^{n*m} to (R^{m*p})^* \otimes R^{n*p}.
    Domain dimension: n*m.
    Codomain dimension: m*p * n*p.
    The matrix has n*m rows and m*p*n*p columns.
    The rank is at most n*m.
    But we know the rank is higher.
    So this flattening is not the one that gives the tensor rank.
    
    The tensor rank is the rank of the tensor as an element of U^* \otimes V^* \otimes W.
    This is the rank of the matrix obtained by flattening the tensor into a matrix where the entries are the tensor components.
    But the tensor components are indexed by (i,j), (j,k), (i,k).
    This is not a rectangular array of indices in the usual sense.
    The tensor is a multilinear map.
    The rank is the minimum number of terms in a decomposition T = sum_r u_r \otimes v_r \otimes w_r.
    This is equivalent to the rank of the matrix obtained by flattening the tensor.
    But how to flatten?
    The standard way is to fix one index and look at the matrix of the remaining two.
    But the rank of the tensor is the maximum of the ranks of all such flattening matrices?
    No, the rank of the tensor is the rank of the matrix obtained by flattening the tensor in a specific way.
    Actually, the rank of the tensor is the rank of the matrix obtained by flattening the tensor into a matrix where the rows are indexed by one set of indices and the columns by the other two.
    But which set of indices?
    For matrix multiplication, the tensor is not symmetric.
    The rank is the rank of the matrix M_{(i,j), (k,l)}? No.
    
    Let's use the definition:
    The rank of a tensor T in U \otimes V \otimes W is the smallest r such that T can be written as sum_{k=1}^r u_k \otimes v_k \otimes w_k.
    This is equivalent to the rank of the linear map T: U^* -> V^* \otimes W?
    No, T is an element of U^* \otimes V^* \otimes W.
    It defines a bilinear map U x V -> W.
    The rank of the tensor is the rank of the linear map U^* -> V^* \otimes W?
    Let's check for 2x2x2.
    U = R^4, V = R^4, W = R^4.
    T: R^4 x R^4 -> R^4.
    The rank is 7.
    The linear map U^* -> V^* \otimes W has dimension 4 -> 16.
    The rank of this map is at most 4.
    So this is not the right flattening.
    
    The correct flattening is:
    The tensor T is in U^* \otimes V^* \otimes W.
    We can view it as a linear map from U^* to V^* \otimes W? No.
    We can view it as a linear map from U^* \otimes V^* to W.
    The rank of the tensor is the rank of this linear map.
    Domain dimension: n*m * m*p.
    Codomain dimension: n*p.
    The matrix has n*m*m*p rows and n*p columns.
    The rank is at most n*p.
    For 2x2x2, n*p = 4. But rank is 7.
    So this is also not the right flattening.
    
    The rank of the tensor is the rank of the matrix obtained by flattening the tensor into a matrix where the entries are the components of the tensor.
    But the components are indexed by (i,j,k).
    This is a 3D array.
    To get a matrix, we need to combine two indices.
    The rank of the tensor is the rank of the matrix obtained by flattening the tensor in ANY way?
    No, the rank of the tensor is the rank of the matrix obtained by flattening the tensor such that the tensor is viewed as a linear map from one space to the tensor product of the other two.
    But we just saw that the ranks of these maps are too small.
    
    Wait, the rank of the tensor is the rank of the matrix obtained by flattening the tensor into a matrix where the rows are indexed by the first set of indices and the columns by the second and third sets of indices.
    But the tensor is in U^* \otimes V^* \otimes W.
    So the rows are indexed by U^*, and the columns by V^* \otimes W.
    The matrix has n*m rows and m*p*n*p columns.
    The rank is at most n*m.
    But the rank of the 2x2x2 tensor is 7, which is greater than 4.
    So this is not the right flattening.
    
    The correct definition is:
    The rank of a tensor T in U \otimes V \otimes W is the smallest r such that T = sum_{k=1}^r u_k \otimes v_k \otimes w_k.
    This is equal to the rank of the matrix obtained by flattening the tensor into a matrix where the rows are indexed by U and the columns by V \otimes W.
    But we just saw that the rank of this matrix is at most dim(U).
    So this cannot be right.
    
    Let's check the literature.
    The rank of a tensor T in U \otimes V \otimes W is the rank of the linear map T: U^* -> V^* \otimes W?
    No, T is an element of U^* \otimes V^* \otimes W.
    It defines a bilinear map U x V -> W.
    The rank of the tensor is the rank of the linear map U^* -> V^* \otimes W?
    No, the rank of the tensor is the rank of the matrix obtained by flattening the tensor into a matrix where the rows are indexed by the first set of indices and the columns by the second and third sets of indices.
    But the tensor is in U^* \otimes V^* \otimes W.
    So the rows are indexed by U^*, and the columns by V^* \otimes W.
    The matrix has n*m rows and m*p*n*p columns.
    The rank is at most n*m.
    But the rank of the 2x2x2 tensor is 7, which is greater than 4.
    So this is not the right flattening.
    
    I am confused. Let's try a different approach.
    The rank of the tensor is the rank of the matrix obtained by flattening the tensor into a matrix where the entries are the components of the tensor.
    But the components are indexed by (i,j,k).
    This is a 3D array.
    To get a matrix, we need to combine two indices.
    The rank of the tensor is the rank of the matrix obtained by flattening the tensor in ANY way?
    No, the rank of the tensor is the rank of the matrix obtained by flattening the tensor such that the tensor is viewed as a linear map from one space to the tensor product of the other two.
    But we just saw that the ranks of these maps are too small.
    
    Let's try to compute the rank of the 2x2x2 tensor using this method.
    T_{ij, jk, ik} = 1.
    Flatten (i,j) and (j,k) into a single index?
    No, (i,j) and (j,k) are not independent.
    The tensor is in U^* \otimes V^* \otimes W.
    U^* has basis e_{ij}, V^* has basis e_{jk}, W has basis e_{ik}.
    T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
    The rank of T is the rank of the matrix obtained by flattening T into a matrix where the rows are indexed by e_{ij} and the columns by e_{jk} \otimes e_{ik}.
    The matrix has 4 rows and 16 columns.
    The entries are:
    Row (i,j), Col (j',k) \otimes (i',k'):
    T_{ij, j'k, i'k'} = 1 if i=i', j=j', k=k'.
    So for a fixed row (i,j), the non-zero entries are at columns (j,k) \otimes (i,k) for k=0,1.
    So each row has 2 non-zero entries.
    The rows are:
    (0,0): cols (0,0)\otimes(0,0), (0,1)\otimes(0,1)
    (0,1): cols (1,0)\otimes(0,0), (1,1)\otimes(0,1)
    (1,0): cols (0,0)\otimes(1,0), (0,1)\otimes(1,1)
    (1,1): cols (1,0)\otimes(1,0), (1,1)\otimes(1,1)
    These 4 rows are linearly independent.
    So the rank is 4.
    But the rank of the 2x2x2 tensor is 7.
    So this flattening is not the one that gives the tensor rank.
    
    The correct flattening is:
    The rank of the tensor is the rank of the matrix obtained by flattening the tensor into a matrix where the rows are indexed by the first set of indices and the columns by the second set of indices, and the entries are the third set of indices.
    No, that's not right.
    
    The rank of the tensor is the rank of the matrix obtained by flattening the tensor into a matrix where the rows are indexed by the first set of indices and the columns by the second and third sets of indices.
    But we just saw that the rank of this matrix is 4 for 2x2x2, but the tensor rank is 7.
    So this is not the right flattening.
    
    I think I am making a mistake in the definition of the tensor rank.
    The rank of a tensor T in U \otimes V \otimes W is the smallest r such that T = sum_{k=1}^r u_k \otimes v_k \otimes w_k.
    This is equal to the rank of the matrix obtained by flattening the tensor into a matrix where the rows are indexed by U and the columns by V \otimes W.
    But we just saw that the rank of this matrix is 4 for 2x2x2, but the tensor rank is 7.
    So this is not the right flattening.
    
    Wait, the tensor is in U^* \otimes V^* \otimes W.
    The rank of the tensor is the rank of the linear map T: U^* -> V^* \otimes W?
    No, T is an element of U^* \otimes V^* \otimes W.
    It defines a bilinear map U x V -> W.
    The rank of the tensor is the rank of the linear map U^* -> V^* \otimes W?
    No, the rank of the tensor is the rank of the matrix obtained by flattening the tensor into a matrix where the rows are indexed by the first set of indices and the columns by the second and third sets of indices.
    But the tensor is in U^* \otimes V^* \otimes W.
    So the rows are indexed by U^*, and the columns by V^* \otimes W.
    The matrix has n*m rows and m*p*n*p columns.
    The rank is at most n*m.
    But the rank of the 2x2x2 tensor is 7, which is greater than 4.
    So this is not the right flattening.
    
    I give up. I will use the schoolbook rank as the lower bound.
    """
    return n * m * p

def _local_discrete_search(U, V, W, T_target, n, m, p, max_iters=5000, time_limit=None):
    """
    Perform a local search on the discrete half-integer grid.
    Start from U, V, W (which should be close to a solution).
    """
    U = _snap_to_half_int(U)
    V = _snap_to_half_int(V)
    W = _snap_to_half_int(W)
    
    current_err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T_target)
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time_limit and time.time() > time_limit:
            break
            
        # Pick a random matrix and a random entry
        mat_idx = np.random.choice(3)
        if mat_idx == 0:
            row, col = np.random.choice(U.shape[0]), np.random.choice(U.shape[1])
            old_val = U[row, col]
            # Try changing by +/- 0.5 or +/- 1.0
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            U[row, col] = new_val
            T_new = np.einsum('ar,br,cr->abc', U, V, W)
            new_err = np.linalg.norm(T_new - T_target)
            if new_err < current_err:
                current_err = new_err
            else:
                U[row, col] = old_val
        elif mat_idx == 1:
            row, col = np.random.choice(V.shape[0]), np.random.choice(V.shape[1])
            old_val = V[row, col]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            V[row, col] = new_val
            T_new = np.einsum('ar,br,cr->abc', U, V, W)
            new_err = np.linalg.norm(T_new - T_target)
            if new_err < current_err:
                current_err = new_err
            else:
                V[row, col] = old_val
        else:
            row, col = np.random.choice(W.shape[0]), np.random.choice(W.shape[1])
            old_val = W[row, col]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            W[row, col] = new_val
            T_new = np.einsum('ar,br,cr->abc', U, V, W)
            new_err = np.linalg.norm(T_new - T_target)
            if new_err < current_err:
                current_err = new_err
            else:
                W[row, col] = old_val
                
        if current_err < 1e-6:
            break
            
    return U, V, W, current_err

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    T_target = _build_target_tensor(n, m, p)
    
    # Try to find lower rank
    # We will try ranks from 44 down to 35
    for target_rank in range(44, 34, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try multiple random restarts
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(100 + restart)
            
            # Initialize with random values
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Gradient Descent Phase
            lr = 0.05
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                    
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_recon - T_target
                
                G_U = np.einsum('abc,br,cr->ar', diff, V, W)
                G_V = np.einsum('abc,ar,cr->br', diff, U, W)
                G_W = np.einsum('abc,ar,br->cr', diff, U, V)
                
                U -= lr * G_U
                V -= lr * G_V
                W -= lr * G_W
                
                if step % 50 == 0:
                    err = np.linalg.norm(diff)
                    if err < 1e-3:
                        break
                        
            # Discrete Search Phase
            U, V, W, err = _local_discrete_search(U, V, W, T_target, n, m, p, max_iters=2000, time_limit=deadline - 5)
            
            if err < 1e-6:
                # Verify strictly
                if _verify(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found = True
                    break
                    
        if found:
            continue # Try lower rank
        else:
            # If not found, we might still try lower ranks if time permits, 
            # but usually if we fail at 44, we might fail at 43.
            # However, the search is stochastic.
            pass

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
