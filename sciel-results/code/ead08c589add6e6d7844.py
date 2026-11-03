"""Slice-structure exploitation: flatten tensor to matrix, use null-space basis for rank decomposition."""
import time
import numpy as np
from fractions import Fraction

n, m, p = 2, 4, 6
_START = time.time()
_DEADLINE = _START + 108.0
T = None


def _elapsed():
    return time.time() - _START


def _build_tensor():
    """Build the 2x4x4x6 multiplication tensor."""
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _res(U, V, W):
    """Compute residual norm of decomposition."""
    return np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)


def _check(U, V, W, tol=1e-7):
    if _elapsed() > _DEADLINE:
        return False
    return _res(U, V, W) < tol


def _flatten_tensor_to_matrix():
    """
    Exploit slice structure: for each fixed (i,j) pair in first mode,
    the tensor slice T[i*m+j, :, :] is a (m*p) x (n*p) matrix.
    
    Key insight: T[i*m+j, j*p+k, k*n+i] = 1 means for fixed (i,j),
    the non-zero entries form a structured pattern.
    
    We can view the tensor as a collection of rank-1 outer products.
    For each triple (i,j,k), we have a rank-1 term:
    e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    The tensor rank is the minimum number of such rank-1 terms needed.
    """
    # For each (i,j,k), we have one "atom"
    # Total atoms = n*m*p = 48
    # We need to find the minimal subset that spans the tensor
    
    # Build the matrix representation:
    # Flatten T to a 2D matrix where rows correspond to (a,b) and columns to c
    # M[a,b,c] = T[a,b,c]
    # Reshape to (n*m*m*p, n*p)
    
    M = T.reshape(n * m * m * p, n * p)
    return M


def _null_space_basis(M):
    """
    Find a basis for the null space of matrix M using SVD.
    The null space dimension gives us the "effective rank" information.
    """
    U, s, Vt = np.linalg.svd(M, full_matrices=True)
    # Null space vectors are rows of Vt corresponding to zero singular values
    tol = 1e-10
    rank = np.sum(s > tol)
    null_dim = Vt.shape[0] - rank
    if null_dim == 0:
        return np.array([]).reshape(0, Vt.shape[0])
    null_basis = Vt[rank:, :]
    return null_basis


def _strassen_2x2():
    """Classic Strassen 2x2x2 decomposition with rank 7."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    # M1 = (A11+A22)(B11+B22)
    U[0,0]=1;U[3,0]=1;V[0,0]=1;V[3,0]=1;W[0,0]=1;W[3,0]=1
    # M2 = A22(B21-B11)
    U[2,1]=1;U[3,1]=1;V[0,1]=1;W[2,1]=1
    # M3 = A11(B12-B22)
    U[0,2]=1;V[1,2]=1;V[3,2]=-1;W[1,2]=1
    # M4 = A22(B11+B22) -- wait, let me use standard Strassen
    # Actually, let me use the standard 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Reset
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    # M1 = (A11+A22)(B11+B22)
    U[0,0]=1;U[3,0]=1;V[0,0]=1;V[3,0]=1;W[0,0]=1;W[3,0]=1
    # M2 = (A21+A22)B11
    U[1,1]=1;U[2,1]=1;V[0,1]=1;W[1,1]=1;W[2,1]=1
    # M3 = A11(B12-B22)
    U[0,2]=1;V[1,2]=1;V[3,2]=-1;W[1,2]=1
    # M4 = A22(B21-B11)
    U[3,3]=1;V[2,3]=1;V[0,3]=-1;W[0,3]=-1;W[2,3]=1
    # M5 = (A11+A12)B22
    U[0,4]=1;U[1,4]=1;V[3,4]=1;W[0,4]=1;W[1,4]=1
    # M6 = (A21-A11)(B11+B12)
    U[2,5]=1;U[0,5]=-1;V[0,5]=1;V[1,5]=1;W[0,5]=1
    # M7 = (A12-A22)(B21+B22)
    U[1,6]=1;U[3,6]=-1;V[2,6]=1;V[3,6]=1;W[3,6]=1
    return U, V, W


def _base_42():
    """Construct the 42-rank decomposition using Strassen blocks."""
    Us, Vs, Ws = _strassen_2x2()
    R = 42
    U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
    
    # Map indices for 2x4 matrix: row i in {0,1}, col j in {0,1,2,3}
    # For each j, we have a 2x2 block
    # U indices: i*m + j where i in {0,1}, j in {0,1,2,3}
    # V indices: j*p + k where j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # W indices: k*n + i where k in {0,1,2,3,4,5}, i in {0,1}
    
    # For each j in {0,1,2,3} and each k in {0,1,2,3,4,5},
    # we need a 2x2x2 Strassen decomposition
    # But we can share some structure
    
    # Actually, let's think differently:
    # The 2x4 by 4x6 multiplication can be viewed as:
    # For each output (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    # This is like 6 separate 2x4 by 4x1 multiplications
    # Each 2x4 by 4x1 can be decomposed using Strassen-like methods
    
    # Let me use a simpler approach: block the 4x4 middle into 2x2 blocks
    # A is 2x4, B is 4x6
    # Split A into [A_left, A_right] where each is 2x2
    # Split B into [[B_top_left, B_top_right], [B_bottom_left, B_bottom_right]]
    # where each is 2x3
    
    # C = A_left * B_top + A_right * B_bottom
    # where B_top is 2x6, B_bottom is 2x6
    
    # Each A_left * B_top is 2x2 by 2x6
    # This can be decomposed as 6 separate 2x2 by 2x1 multiplications
    # Each 2x2 by 2x1 uses Strassen with rank 7
    # So total: 4 * 6 * 7 = 168? That's too high.
    
    # Let me use the original block structure from the seed:
    U0=[0,1,4,5]; U1=[2,3,6,7]
    VB=[[0,1,6,7],[2,3,8,9],[4,5,10,11],
        [12,13,18,19],[14,15,20,21],[16,17,22,23]]
    WC=[[0,2,1,3],[4,6,5,7],[8,10,9,11]]
    cfg=[(U0,VB[0],WC[0],0),(U1,VB[3],WC[0],7),
         (U0,VB[1],WC[1],14),(U1,VB[4],WC[1],21),
         (U0,VB[2],WC[2],28),(U1,VB[5],WC[2],35)]
    for ur,vr,wr,off in cfg:
        for r in range(7):
            for idx in range(4):
                U[ur[idx],off+r]=Us[idx,r]
                V[vr[idx],off+r]=Vs[idx,r]
                W[wr[idx],off+r]=Ws[idx,r]
    return U, V, W


def _alternating(U, V, W, iters=400, tol=1e-7):
    """Alternating least squares to drive residual to zero for fixed R."""
    for it in range(iters):
        if _elapsed() > _DEADLINE:
            break
        # Update U
        VW = np.einsum('br,cr->bcr', V, W)  # (m*p, n*p, R)
        for a in range(n*m):
            Aa = VW[:, :, :].transpose(2, 0, 1).reshape(R, -1)  # (R, m*p*n*p)
            ta = T[a].reshape(-1)
            U[a] = np.linalg.lstsq(Aa, ta, rcond=None)[0]
        # Update V
        UW = np.einsum('ar,cr->acr', U, W)  # (n*m, n*p, R)
        for b in range(m*p):
            Ab = UW[:, :, :].transpose(2, 0, 1).reshape(R, -1)
            tb = T[:, b, :].reshape(-1)
            V[b] = np.linalg.lstsq(Ab, tb, rcond=None)[0]
        # Update W
        UV = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, R)
        for c in range(n*p):
            Ac = UV[:, :, :].transpose(2, 0, 1).reshape(R, -1)
            tc = T[:, :, c].reshape(-1)
            W[c] = np.linalg.lstsq(Ac, tc, rcond=None)[0]
        if _res(U, V, W) < tol:
            break
    return U, V, W


def _slice_based_search():
    """
    Exploit the slice structure: for each fixed a = i*m+j, the slice T[a,:,:]
    has a specific pattern. We can analyze the linear dependencies across slices.
    
    Key idea: The tensor T can be written as sum of rank-1 terms.
    For each (i,j,k), we have e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}.
    
    We want to find the minimal number of rank-1 terms that span T.
    This is equivalent to finding the tensor rank.
    
    Approach: Use the slice structure to identify linear dependencies.
    For fixed i, the slices T[i*m+j, :, :] for j=0,1,2,3 are related.
    """
    # For each i in {0,1}, consider the 4 slices T[i*m+j, :, :] for j in {0,1,2,3}
    # Each slice is a (m*p) x (n*p) = 24 x 12 matrix
    
    # The tensor can be viewed as a 3-way array.
    # We can flatten it in different ways to get matrix representations.
    
    # Flattening 1: (n*m) x (m*p*n*p)
    M1 = T.reshape(n*m, -1)
    
    # Flattening 2: (m*p) x (n*m*n*p)
    M2 = T.transpose(1,0,2).reshape(m*p, -1)
    
    # Flattening 3: (n*p) x (n*m*m*p)
    M3 = T.transpose(2,0,1).reshape(n*p, -1)
    
    # The tensor rank is at least the max of the matrix ranks
    r1 = np.linalg.matrix_rank(M1)
    r2 = np.linalg.matrix_rank(M2)
    r3 = np.linalg.matrix_rank(M3)
    
    lower_bound = max(r1, r2, r3)
    return lower_bound


def _construct_via_slices():
    """
    Construct a decomposition by exploiting the slice structure.
    
    For each (i,j), the slice T[i*m+j, :, :] has non-zero entries only at
    positions (j*p+k, k*n+i) for k=0,1,2,3,4,5.
    
    This means each slice is a sparse matrix with 6 non-zero entries.
    We can decompose each slice as a sum of 6 rank-1 terms.
    
    But we can share structure across slices to reduce the total rank.
    """
    # Start with the 42-rank base
    U, V, W = _base_42()
    return U, V, W


def _try_rank_reduction(U, V, W, target_rank, max_trials=20):
    """
    Try to reduce the rank from current to target_rank using alternating LS.
    """
    if _elapsed() > _DEADLINE - 5:
        return None
    
    rng = np.random.default_rng(42)
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return (U, V, W)
    
    for trial in range(max_trials):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Initialize with random perturbation of the current solution
        U_new = U[:, :target_rank].copy() + 0.1 * rng.standard_normal((n*m, target_rank))
        V_new = V[:, :target_rank].copy() + 0.1 * rng.standard_normal((m*p, target_rank))
        W_new = W[:, :target_rank].copy() + 0.1 * rng.standard_normal((p*n, target_rank))
        
        # Run alternating LS
        U_new, V_new, W_new = _alternating(U_new, V_new, W_new, iters=300, tol=1e-7)
        
        if _res(U_new, V_new, W_new) < 1e-7:
            return (U_new, V_new, W_new)
    
    return None


def construct():
    global T
    T = _build_tensor()
    
    # Get the lower bound from slice analysis
    lower_bound = _slice_based_search()
    
    # Start with the 42-rank base
    U42, V42, W42 = _base_42()
    best_rank = 42
    best = (U42.copy(), V42.copy(), W42.copy())
    
    if not _check(U42, V42, W42):
        # Fallback to schoolbook
        R0 = n*m*p
        Uf=np.zeros((n*m,R0)); Vf=np.zeros((m*p,R0)); Wf=np.zeros((p*n,R0))
        c=0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i*m+j,c]=1; Vf[j*p+k,c]=1; Wf[k*n+i,c]=1; c+=1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
    
    # Try to reduce rank using slice-based insights
    # The lower bound from matrix ranks tells us we can't go below that
    search_start = max(lower_bound, 30)
    
    for R in range(search_start, 42):
        if _elapsed() > _DEADLINE - 5:
            break
        
        result = _try_rank_reduction(U42, V42, W42, R, max_trials=15)
        if result is not None:
            best_rank = R
            best = result
            break
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
