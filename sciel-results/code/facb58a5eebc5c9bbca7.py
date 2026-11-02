"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    deadline = time.time() + 140
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Precompute: for each (i,j,k), the flattened indices
    eqs = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    idx_U = np.array([i*m+j for i, j, k in eqs])
    idx_V = np.array([j*p+k for i, j, k in eqs])
    idx_W = np.array([k*n+i for i, j, k in eqs])
    
    def max_err(U, V, W):
        contrib = U[idx_U, :] * V[idx_V, :] * W[idx_W, :]
        return np.max(np.abs(contrib.sum(axis=1) - 1.0))
    
    # Try to find rank < 64
    # Use local search from schoolbook, trying to merge columns
    best = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_school
    
    # Start with schoolbook and try to reduce
    U = np.array(U_fb, dtype=np.float64)
    V = np.array(V_fb, dtype=np.float64)
    W = np.array(W_fb, dtype=np.float64)
    
    # Try setting some columns to zero and compensating
    # This is complex; instead try random low-rank init + search
    
    for rank in range(40, rank_school):
        if time.time() > deadline - 3:
            break
        for restart in range(30):
            if time.time() > deadline - 3:
                break
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for c in range(rank):
                # Random sparse init
                nz_U = np.random.choice(n*m, np.random.randint(1,5), replace=False)
                U[nz_U, c] = np.random.choice([-1,1], len(nz_U))
                nz_V = np.random.choice(m*p, np.random.randint(1,5), replace=False)
                V[nz_V, c] = np.random.choice([-1,1], len(nz_V))
                nz_W = np.random.choice(p*n, np.random.randint(1,5), replace=False)
                W[nz_W, c] = np.random.choice([-1,1], len(nz_W))
            
            for it in range(300):
                if time.time() > deadline - 3:
                    break
                contrib = U[idx_U, :] * V[idx_V, :] * W[idx_W, :]
                total = contrib.sum(axis=1)
                errs = np.abs(total - 1.0)
                n_err = np.sum(errs > 1e-9)
                if n_err == 0:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                # Fix a random error
                ei = np.where(errs > 1e-9)[0]
                eq = ei[np.random.randint(len(ei))]
                i, j, k = eqs[eq]
                c = np.random.randint(rank)
                u,v,w = U[i*m+j,c], V[j*p+k,c], W[k*n+i,c]
                other = total[eq] - u*v*w
                if v!=0 and w!=0:
                    old=U[i*m+j,c]; U[i*m+j,c]=(1-other)/(v*w)
                    if np.sum(np.abs((U[idx_U,:]*V[idx_V,:]*W[idx_W,:]).sum(axis=1)-1.0)>1e-9)>n_err: U[i*m+j,c]=old
                elif u!=0 and w!=0:
                    old=V[j*p+k,c]; V[j*p+k,c]=(1-other)/(u*w)
                    if np.sum(np.abs((U[idx_U,:]*V[idx_V,:]*W[idx_W,:]).sum(axis=1)-1.0)>1e-9)>n_err: V[j*p+k,c]=old
                elif u!=0 and v!=0:
                    old=W[k*n+i,c]; W[k*n+i,c]=(1-other)/(u*v)
                    if np.sum(np.abs((U[idx_U,:]*V[idx_V,:]*W[idx_W,:]).sum(axis=1)-1.0)>1e-9)>n_err: W[k*n+i,c]=old
                elif u==0 and v==0:
                    U[i*m+j,c]=1; V[j*p+k,c]=(1-other)/w if w!=0 else 0
                elif u==0 and w==0:
                    U[i*m+j,c]=1; W[k*n+i,c]=(1-other)/v if v!=0 else 0
                elif v==0 and w==0:
                    V[j*p+k,c]=1; W[k*n+i,c]=(1-other)/u if u!=0 else 0
    
    return best
# EVOLVE-BLOCK-END
