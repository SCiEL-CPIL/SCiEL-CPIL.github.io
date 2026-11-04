import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def verify(U, V, W, rank):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # ---- Strassen 2x2 * 2x2 (rank 7) ----
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[0,2]=1; W2[1,2]=1
    U2[0,3]=1; U2[3,3]=-1; V2[1,3]=1; V2[3,3]=1; W2[0,3]=1; W2[3,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[1,4]=1; W2[3,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[2,5]=1; W2[0,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[1,6]=1; W2[2,6]=1

    # ---- 4x4 * 4x4 via Kronecker Strassen (rank 49) ----
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    for t1 in range(7):
        for t2 in range(7):
            col = t1*7 + t2
            for i1 in range(2):
                if U2[i1,t1]!=0:
                    for i2 in range(2):
                        if U2[i2,t2]!=0:
                            U44[i1*2+i2, col] = U2[i1,t1]*U2[i2,t2]
            for i1 in range(2):
                if V2[i1,t1]!=0:
                    for i2 in range(2):
                        if V2[i2,t2]!=0:
                            V44[i1*2+i2, col] = V2[i1,t1]*V2[i2,t2]
            for i1 in range(2):
                if W2[i1,t1]!=0:
                    for i2 in range(2):
                        if W2[i2,t2]!=0:
                            W44[i1*2+i2, col] = W2[i1,t1]*W2[i2,t2]

    # ---- 4x4 * 4x1 (rank 4) ----
    U41 = np.zeros((16, 4))
    V41 = np.zeros((20, 4))
    W41 = np.zeros((20, 4))
    for r in range(4):
        for i in range(4):
            for j in range(4):
                if i == r:
                    U41[i*4+j, r] = 1.0
                    W41[16+i, r] = 1.0
        for j in range(4):
            V41[j*5+4, r] = 1.0

    # ---- Combine into 4x4 * 4x5 (rank 53) ----
    rank = 53
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    U[:, 0:49] = U44
    for j in range(4):
        for k in range(4):
            V[j*5+k, 0:49] = V44[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W44[k*4+i, 0:49]
    U[:, 49:53] = U41
    V[:, 49:53] = V41
    W[:, 49:53] = W41

    if not verify(U, V, W, rank):
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j,col]=1.0; V[j*p+k,col]=1.0; W[k*n+i,col]=1.0; col+=1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Refinement: try to reduce rank ----
    def residual(U, V, W, rank):
        # smooth surrogate: sum of squared residuals over all (i,j,k)
        r = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    r += (val - 1.0)**2
        return r

    # Try to drop columns that are redundant.
    # For a column c, its contribution to tensor is outer(U[:,c], V[:,c], W[:,c]).
    # If removing it still yields exact tensor, drop it.
    def try_drop_columns(U, V, W, rank):
        # iterate: attempt to drop each column
        changed = True
        while changed:
            changed = False
            for c in range(rank-1, -1, -1):
                if time.time() > deadline:
                    return U, V, W, rank
                # build without column c
                mask = np.ones(rank, dtype=bool); mask[c] = False
                Uc = U[:, mask]; Vc = V[:, mask]; Wc = W[:, mask]
                if verify(Uc, Vc, Wc, rank-1):
                    U, V, W = Uc, Vc, Wc
                    rank -= 1
                    changed = True
        return U, V, W, rank

    U, V, W, rank = try_drop_columns(U, V, W, rank)

    # Systematic column elimination via linear algebra.
    # For each column c, check if its tensor contribution is in the span of others.
    # The contribution of column c is the outer product U[:,c] ⊗ V[:,c] ⊗ W[:,c].
    # We can flatten this into a vector and check if it's in the span of flattened
    # contributions of other columns.
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    def flatten_contribution(Uc, Vc, Wc):
        """Flatten the tensor contribution of a single column."""
        return np.outer(np.outer(Uc, Vc), Wc).ravel()

    def try_systematic_elimination(U, V, W, rank):
        """Try to eliminate columns by checking linear dependence of contributions."""
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                break
            for c in range(rank-1, -1, -1):
                if time.time() > deadline:
                    break
                # Build the flattened contribution of column c
                target_vec = flatten_contribution(U[:,c], V[:,c], W[:,c])
                
                # Build the matrix of flattened contributions of all other columns
                other_cols = [i for i in range(rank) if i != c]
                if len(other_cols) == 0:
                    continue
                    
                # Build matrix where each row is the flattened contribution of an other column
                # Shape: (len(other_cols), n*m*p)
                contrib_matrix = np.array([
                    flatten_contribution(U[:,col], V[:,col], W[:,col]) 
                    for col in other_cols
                ])
                
                # Check if target_vec is in the column space of contrib_matrix.T
                # i.e., is target_vec in the row space of contrib_matrix?
                # Solve: contrib_matrix @ x = target_vec
                try:
                    # Use least squares to check if target is in span
                    x, residuals, rank_m, sv = np.linalg.lstsq(contrib_matrix, target_vec, rcond=1e-10)
                    # Check if the reconstruction is close enough
                    recon = contrib_matrix @ x
                    if np.linalg.norm(recon - target_vec) < 1e-8 * (1 + np.linalg.norm(target_vec)):
                        # Column c is redundant
                        mask = np.ones(rank, dtype=bool); mask[c] = False
                        U = U[:, mask]; V = V[:, mask]; W = W[:, mask]
                        rank -= 1
                        changed = True
                        break  # restart the loop with new rank
                except np.linalg.LinAlgError:
                    continue
        return U, V, W, rank

    # First, try systematic elimination
    U, V, W, rank = try_systematic_elimination(U, V, W, rank)
    if rank < best_rank:
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Final drop pass with verification
    U, V, W, rank = try_drop_columns(U, V, W, rank)

    # Try systematic elimination again after drop pass
    U, V, W, rank = try_systematic_elimination(U, V, W, rank)

    if rank < best_rank:
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if rank >= best_rank:
        U, V, W, rank = best_U, best_V, best_W, best_rank

    if not verify(U, V, W, rank):
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j,col]=1.0; V[j*p+k,col]=1.0; W[k*n+i,col]=1.0; col+=1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
