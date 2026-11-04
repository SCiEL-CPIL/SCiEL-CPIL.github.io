"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _construct_target_tensor(n, m, p):
    """Construct the target multiplication tensor T."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, T, tol=1e-8):
    """
    Verify if U, V, W decompose T.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _try_rational_snapping(U, V, W, T, max_denom=8):
    """
    Try to snap the float solution to rational numbers with small denominators.
    Returns True if a snapped version verifies exactly, and updates U, V, W in place.
    """
    for denom in [1, 2, 4, 8]:
        step = 1.0 / denom
        U_snap = np.round(U / step) * step
        V_snap = np.round(V / step) * step
        W_snap = np.round(W / step) * step
        
        if _verify_fast(U_snap, V_snap, W_snap, T, tol=1e-9):
            U[:] = U_snap
            V[:] = V_snap
            W[:] = W_snap
            return True
            
    return False

def _strassen_2x2_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    where indices are for 2x2 matrices.
    """
    # Strassen's 7 multiplications for 2x2 matmul
    # C[0,0] = M1 - M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 - M6
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U[i,j,r] for A indices (i,j), V[j,k,r] for B indices (j,k), W[k,i,r] for C indices (k,i)
    # Using 0-based indexing: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    U = np.zeros((4, 7))  # 4 A-entries x 7 rank terms
    V = np.zeros((4, 7))  # 4 B-entries x 7 rank terms
    W = np.zeros((4, 7))  # 4 C-entries x 7 rank terms
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = -1 # C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1  # B10
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    
    # M4 = A11(B01-B10)
    U[3, 3] = 1  # A11
    V[1, 3] = 1  # B01
    V[2, 3] = -1 # B10
    W[0, 3] = -1 # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = -1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    W[3, 6] = 1  # C11
    
    return U, V, W

def _build_strassen_based_decomposition(n, m, p):
    """
    Build a decomposition for n x m x p matrix multiplication using Strassen's method
    on overlapping 2x2 blocks.
    
    For 2x5x6: We can partition the 5x6 matrix B into blocks and use Strassen
    on the 2x2 subproblems where applicable.
    
    Strategy: Use the fact that we can decompose the multiplication by considering
    how 2x5 and 5x6 matrices interact. We'll use a combination of direct terms
    and Strassen-based reductions.
    
    For 2x5x6, a good approach is to note that we can split the 5 columns of B into
    groups and apply Strassen where we have 2x2 sub-blocks.
    
    Let's use a more systematic approach:
    - The 2x5 matrix A has 10 entries
    - The 5x6 matrix B has 30 entries  
    - The 2x6 result C has 12 entries
    
    We can think of this as computing C = A @ B where A is 2x5 and B is 5x6.
    
    One approach: Split B into B1 (5x2) and B2 (5x4), or use other partitions.
    Actually, let's use a different strategy based on the structure.
    
    For the 2x5x6 case, we can use the following:
    - Compute partial products for each column of B
    - Use Strassen-like tricks to reduce the number of multiplications
    
    Let me try a concrete construction:
    Split the 5 middle dimension into 2+3 or use overlapping windows.
    
    Actually, the simplest good construction: 
    Use rank-7 Strassen for each 2x2 block where possible, and direct for the rest.
    
    For 2x5x6: 
    - We have 2 rows in A, 5 cols in A/rows in B, 6 cols in B
    - Think of B as having 6 columns, each of length 5
    - C[0,k] = sum_j A[0,j]*B[j,k]
    - C[1,k] = sum_j A[1,j]*B[j,k]
    
    For each pair of output columns (k1, k2), we could potentially use Strassen
    if we group the A entries appropriately.
    
    Let me try: partition the 5 middle indices into {0,1} and {2,3,4}
    Or better: use a known result. The rank of 2x5x6 matmul is at most 54.
    
    Here's a construction:
    Split the 5x6 B matrix into two 5x3 blocks: B = [B_left, B_right]
    Then C = A @ B = [A@B_left, A@B_right]
    Each A@B_block is 2x5 times 5x3 = 2x3, which has rank at most 6 (schoolbook) 
    or we can do better.
    
    Actually, let's just use the direct construction with some Strassen optimization.
    
    For 2x5x6, schoolbook gives rank 60. 
    Using Strassen on 2x2 subproblems:
    
    Consider splitting the 5 middle dimension as 2+3:
    A = [A1 | A2] where A1 is 2x2, A2 is 2x3
    B = [B1; B2] where B1 is 2x6, B2 is 3x6
    C = A1@B1 + A2@B2
    
    A1@B1 is 2x2 @ 2x6: can use Strassen on the 2x2 part with 6 outputs
    A2@B2 is 2x3 @ 3x6: schoolbook rank 18
    
    For 2x2 @ 2x6: Think of B1 as three 2x2 blocks: B1 = [B1a, B1b, B1c]
    Each 2x2 @ 2x2 uses rank 7 (Strassen)
    So 2x2 @ 2x6 uses rank 21
    
    Total: 21 + 18 = 39? That seems too good. Let me reconsider.
    
    Actually, for 2x2 @ 2x6, we compute C = A1 @ B1 where A1 is 2x2, B1 is 2x6.
    C[0,0] = A1[0,0]*B1[0,0] + A1[0,1]*B1[1,0]
    C[0,1] = A1[0,0]*B1[0,1] + A1[0,1]*B1[1,1]
    ... etc
    
    This is like computing 3 separate 2x2 @ 2x2 multiplications but sharing A1.
    Since A1 is shared, we can't simply use 3*7=21.
    
    Let me use a different approach. I'll construct a valid rank-54 decomposition
    based on known results and verify it.
    
    For 2x5x6, one known upper bound construction:
    Use the fact that rank(2x5x6) <= rank(2x5x5) + rank(2x5x1)
    rank(2x5x5) <= ? and rank(2x5x1) = 10 (each C[i,k] = sum_j A[i,j]*B[j,0], 2*5=10 terms)
    
    Actually, let me just implement a search that starts from the schoolbook solution
    and tries to reduce the rank using optimization, with the Strassen insight as
    a guide for initialization.
    
    For now, let's return a valid rank-60 schoolbook solution as fallback,
    and the main optimization will try to find lower ranks.
    """
    n, m, p = 2, 5, 6
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition (rank 60)
    U = np.zeros((nm, nm*mp//n))  # This is wrong, let me just do it properly
    rank_sb = n * m * p
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W, rank_sb

def _als_polish(U, V, W, T, n, m, p, iterations=20):
    """
    Polish the solution using Alternating Least Squares.
    """
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    for _ in range(iterations):
        # Solve for U
        T_mat = T.reshape(nm, mp*pn)
        
        V_mat = np.zeros((mp*pn, r))
        for b in range(mp):
            for c in range(pn):
                V_mat[b*pn+c, :] = V[b, :] * W[c, :]
                
        try:
            VtV = V_mat.T @ V_mat
            VtV += 1e-6 * np.eye(r)
            U = T_mat @ V_mat @ np.linalg.inv(VtV)
        except np.linalg.LinAlgError:
            pass
            
        # Solve for V
        T_mat_v = T.reshape(mp, nm*pn)
        U_mat = np.zeros((nm*pn, r))
        for a in range(nm):
            for c in range(pn):
                U_mat[a*pn+c, :] = U[a, :] * W[c, :]
        try:
            UtU = U_mat.T @ U_mat
            UtU += 1e-6 * np.eye(r)
            V = T_mat_v @ U_mat @ np.linalg.inv(UtU)
        except np.linalg.LinAlgError:
            pass
            
        # Solve for W
        T_mat_w = T.reshape(pn, nm*mp)
        UW_mat = np.zeros((nm*mp, r))
        for a in range(nm):
            for b in range(mp):
                UW_mat[a*mp+b, :] = U[a, :] * V[b, :]
        try:
            UWtUW = UW_mat.T @ UW_mat
            UWtUW += 1e-6 * np.eye(r)
            W = T_mat_w @ UW_mat @ np.linalg.inv(UWtUW)
        except np.linalg.LinAlgError:
            pass

    return U, V, W

def _gradient_descent_with_restart(U, V, W, T, n, m, p, rank, deadline, max_steps=500):
    """
    Run gradient descent with adaptive learning rate and early stopping.
    Returns (found, U, V, W) where found is True if a valid decomposition was found.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    v = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
    m_ = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
    beta1, beta2 = 0.9, 0.999
    lr = 0.02
    t = 0
    best_loss = np.inf
    found = False
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
            
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - recon
        loss = np.sum(R**2)
        
        if loss < best_loss:
            best_loss = loss
            if loss < 1e-10:
                found = True
                break
                
        # Gradients
        gU = -np.einsum('abc,br,cr->ar', R, V, W)
        gV = -np.einsum('abc,ar,cr->br', R, U, W)
        gW = -np.einsum('abc,ar,br->cr', R, U, V)
        
        grads = [gU, gV, gW]
        params = [U, V, W]
        
        t += 1
        for i in range(3):
            m_[i] = beta1 * m_[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
            m_hat = m_[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
        U = params[0]
        V = params[1]
        W = params[2]
        
        # Adaptive learning rate: decay when loss plateaus
        if step % 100 == 0 and step > 0:
            if best_loss > 1e-6:
                lr *= 0.7
            else:
                lr *= 0.5
        
        # Check for convergence
        if step % 20 == 0 and loss < 1e-8:
            if _verify_fast(U, V, W, T, tol=1e-9):
                found = True
                break
                
    return found, U, V, W

def _strassen_init_2x5x6(rank, seed):
    """
    Initialize using Strassen-inspired structure for 2x5x6.
    Uses the rank-7 Strassen decomposition for 2x2 blocks.
    """
    n, m, p = 2, 5, 6
    nm, mp, pn = n*m, m*p, p*n
    np.random.seed(seed)
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Split the 5 middle indices into groups
    # Group 1: indices 0,1 (2x2 Strassen block)
    # Group 2: indices 2,3 (2x2 Strassen block)  
    # Group 3: index 4 (single)
    
    # For each group of 2 middle indices and each pair of output columns,
    # apply Strassen's 7-term decomposition
    
    # Output columns: 0,1,2,3,4,5
    # Pair them: (0,1), (2,3), (4,5)
    
    col_pairs = [(0, 1), (2, 3), (4, 5)]
    mid_groups = [(0, 1), (2, 3)]  # Two 2x2 blocks, one leftover
    
    col_idx = 0
    for mid_g in mid_groups:
        for col_p in col_pairs:
            j1, j2 = mid_g
            k1, k2 = col_p
            
            # Apply Strassen's 7 terms for A[:, j1:j2] @ B[j1:j2, k1:k2]
            # This is a 2x2 @ 2x2 multiplication
            # A entries: A[0,j1], A[0,j2], A[1,j1], A[1,j2]
            # B entries: B[j1,k1], B[j1,k2], B[j2,k1], B[j2,k2]
            # C entries: C[0,k1], C[0,k2], C[1,k1], C[1,k2]
            
            # Strassen terms
            # M1 = (A0j1+A1j2)(Bj1k1+Bj2k2)
            if col_idx < rank:
                U[0*m+j1, col_idx] = 1  # A[0,j1]
                U[1*m+j2, col_idx] = 1  # A[1,j2]
                V[j1*p+k1, col_idx] = 1  # B[j1,k1]
                V[j2*p+k2, col_idx] = 1  # B[j2,k2]
                W[k1*n+0, col_idx] = 1  # C[0,k1]
                W[k2*n+1, col_idx] = -1 # C[1,k2]
                col_idx += 1
            
            # M2 = (A1j1+A1j2)Bj1k1
            if col_idx < rank:
                U[1*m+j1, col_idx] = 1  # A[1,j1]
                U[1*m+j2, col_idx] = 1  # A[1,j2]
                V[j1*p+k1, col_idx] = 1  # B[j1,k1]
                W[k1*n+1, col_idx] = 1  # C[1,k1]
                col_idx += 1
            
            # M3 = A0j1(Bj2k1-Bj2k2)
            if col_idx < rank:
                U[0*m+j1, col_idx] = 1  # A[0,j1]
                V[j2*p+k1, col_idx] = 1  # B[j2,k1]
                V[j2*p+k2, col_idx] = -1 # B[j2,k2]
                W[k2*n+0, col_idx] = 1  # C[0,k2]
                col_idx += 1
            
            # M4 = A1j2(Bj1k2-Bj2k1)
            if col_idx < rank:
                U[1*m+j2, col_idx] = 1  # A[1,j2]
                V[j1*p+k2, col_idx] = 1  # B[j1,k2]
                V[j2*p+k1, col_idx] = -1 # B[j2,k1]
                W[k1*n+0, col_idx] = -1 # C[0,k1]
                W[k1*n+1, col_idx] = 1  # C[1,k1]
                col_idx += 1
            
            # M5 = (A0j1+A0j2)Bj2k2
            if col_idx < rank:
                U[0*m+j1, col_idx] = 1  # A[0,j1]
                U[0*m+j2, col_idx] = 1  # A[0,j2]
                V[j2*p+k2, col_idx] = 1  # B[j2,k2]
                W[k1*n+0, col_idx] = -1 # C[0,k1]
                W[k2*n+0, col_idx] = 1  # C[0,k2]
                col_idx += 1
            
            # M6 = (A1j1-A0j1)(Bj1k1+Bj1k2)
            if col_idx < rank:
                U[1*m+j1, col_idx] = 1  # A[1,j1]
                U[0*m+j1, col_idx] = -1 # A[0,j1]
                V[j1*p+k1, col_idx] = 1  # B[j1,k1]
                V[j1*p+k2, col_idx] = 1  # B[j1,k2]
                W[k2*n+1, col_idx] = -1 # C[1,k2]
                col_idx += 1
            
            # M7 = (A0j2-A1j2)(Bj2k1+Bj2k2)
            if col_idx < rank:
                U[0*m+j2, col_idx] = 1  # A[0,j2]
                U[1*m+j2, col_idx] = -1 # A[1,j2]
                V[j2*p+k1, col_idx] = 1  # B[j2,k1]
                V[j2*p+k2, col_idx] = 1  # B[j2,k2]
                W[k1*n+0, col_idx] = 1  # C[0,k1]
                W[k2*n+1, col_idx] = 1  # C[1,k2]
                col_idx += 1
    
    # Handle the leftover middle index j=4
    # For j=4, we need: C[i,k] += A[i,4]*B[4,k] for i=0,1 and k=0..5
    # This is 2*6=12 terms, but we can share
    for k in range(p):
        if col_idx < rank:
            U[0*m+4, col_idx] = 1  # A[0,4]
            V[4*p+k, col_idx] = 1  # B[4,k]
            W[k*n+0, col_idx] = 1  # C[0,k]
            col_idx += 1
        if col_idx < rank:
            U[1*m+4, col_idx] = 1  # A[1,4]
            V[4*p+k, col_idx] = 1  # B[4,k]
            W[k*n+1, col_idx] = 1  # C[1,k]
            col_idx += 1
    
    # If we still have capacity, add random initialization for remaining columns
    for r in range(col_idx, rank):
        U[:, r] = np.random.randn(nm) * 0.1
        V[:, r] = np.random.randn(mp) * 0.1
        W[:, r] = np.random.randn(pn) * 0.1
    
    return U, V, W

def _structured_init(n, m, p, rank, seed):
    """
    Generate structured initializations based on the tensor's inherent structure.
    For 2x5x6, exploit the small n=2 dimension.
    """
    nm, mp, pn = n*m, m*p, p*n
    np.random.seed(seed)
    
    # Use Strassen-inspired initialization for even seeds
    if seed % 2 == 0:
        return _strassen_init_2x5x6(rank, seed)
    
    # Strategy: Sparse initialization - each rank term activates a few entries
    if seed % 2 == 1:
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for r in range(rank):
            i1 = np.random.randint(0, n)
            i2 = np.random.randint(0, n) if np.random.rand() > 0.5 else i1
            js = np.random.choice(m, size=min(m, np.random.randint(1, m+1)), replace=False)
            ks = np.random.choice(p, size=min(p, np.random.randint(1, p+1)), replace=False)
            
            for j in js:
                for k in ks:
                    U[i1*m+j, r] = np.random.choice([-1, 1])
                    V[j*p+k, r] = np.random.choice([-1, 1])
                    W[k*n+i1, r] = np.random.choice([-1, 1])
                    if i2 != i1:
                        U[i2*m+j, r] = np.random.choice([-1, 1]) * 0.5
                        W[k*n+i2, r] = np.random.choice([-1, 1]) * 0.5
    
    # Strategy: Random initialization
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = _construct_target_tensor(n, m, p)
    
    # Schoolbook fallback (Rank 60)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # First, try to verify and use Strassen-based rank-54 decomposition
    # The Strassen construction for 2x5x6 should give rank 54
    # Let's try ranks from 54 down
    
    # Try rank 54 first with Strassen initialization
    for rank in [54, 53, 52, 51, 50]:
        if time.time() > deadline - 30:
            break
            
        found = False
        max_restarts = 10
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Use Strassen-inspired initialization
            U, V, W = _structured_init(n, m, p, rank, seed=restart * 13 + 7)
            
            # Run gradient descent
            found, U, V, W = _gradient_descent_with_restart(U, V, W, T, n, m, p, rank, deadline, max_steps=300)
            
            if found:
                # Polish with ALS
                U, V, W = _als_polish(U, V, W, T, n, m, p, iterations=15)
                if _verify_fast(U, V, W, T, tol=1e-8):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    _try_rational_snapping(U, V, W, T)
                    break

        if found:
            break
    
    # If we haven't found anything better, try a few more ranks
    if best_rank == rank_sb:
        for rank in range(55, rank_sb):
            if time.time() > deadline - 20:
                break
                
            found = False
            for restart in range(5):
                if time.time() > deadline - 10:
                    break
                    
                U, V, W = _structured_init(n, m, p, rank, seed=100 + restart * 11)
                found, U, V, W = _gradient_descent_with_restart(U, V, W, T, n, m, p, rank, deadline, max_steps=200)
                
                if found:
                    U, V, W = _als_polish(U, V, W, T, n, m, p, iterations=10)
                    if _verify_fast(U, V, W, T, tol=1e-8):
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                        _try_rational_snapping(U, V, W, T)
                        break
            
            if found:
                break

    U_final = best_U
    V_final = best_V
    W_final = best_W
    
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
