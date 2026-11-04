"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    N, M, P = n * m, m * p, p * n  # 10, 30, 12
    schoolbook_rank = n * m * p  # 60
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Key insight: if we set W[k*n+i,c] = 1 for all k,i,c, then:
    # sum_c U[i*m+j,c] * V[j*p+k,c] = 1 for all i,j,k
    # This means U[:,c] * V[:,c] (element-wise in the right sense) sums to 1
    
    # Even simpler: if U[i*m+j,c] = a_c and V[j*p+k,c] = b_c for all i,j,k:
    # sum_c a_c * b_c = 1
    # But U and V can't be constant across all indices if we want to use fewer terms...
    
    # Let's try: rank = m * p = 30
    # Set c = (j,k) pairs
    # U[i*m+j, (j',k')] = delta(j,j') * something
    # V[j*p+k, (j',k')] = delta(k,k') * something
    # W[k*n+i, (j',k')] = something
    # sum_{j',k'} delta(j,j') * delta(k,k') * U[i*m+j,(j,k)] * V[j*p+k,(j,k)] * W[k*n+i,(j,k)] = 1
    # = U[i*m+j,(j,k)] * V[j*p+k,(j,k)] * W[k*n+i,(j,k)] = 1
    # So set U[i*m+j,(j,k)] = 1/sqrt(3), V[j*p+k,(j,k)] = 1/sqrt(3), W[k*n+i,(j,k)] = 3
    # Then product = 1. But this uses rank 30.
    
    # Can we do better? Try rank = m * n = 10
    # c = (i,j) pairs
    # U[i*m+j, (i',j')] = delta(i,i') * delta(j,j')
    # V[j*p+k, (i',j')] = delta(j,j') * something(i',k)
    # W[k*n+i, (i',j')] = delta(i,i') * something(j',k)
    # sum_{i',j'} delta(i,i')delta(j,j') * delta(j,j') * f(i',k) * delta(i,i') * g(j',k)
    # = f(i,k) * g(j,k)
    # Need f(i,k) * g(j,k) = 1 for all i,j,k
    # This requires f(i,k) = 1/g(j,k) for all j, which means f(i,k) is independent of j
    # and g(j,k) is independent of i. So f(i,k) * g(j,k) = 1 means
    # f(i,k) = a(k), g(j,k) = 1/a(k) for some function a(k).
    # Then V[j*p+k, (i',j')] = delta(j,j') * a(k)
    # W[k*n+i, (i',j')] = delta(i,i') * 1/a(k)
    # U[i*m+j, (i',j')] = delta(i,i') * delta(j,j')
    # Check: sum_{i',j'} delta(i,i')delta(j,j') * delta(j,j')a(k) * delta(i,i')/a(k) = 1. Yes!
    # So rank = m*n = 10 works!
    
    # Wait, let me double-check the indices.
    # c indexes (i', j') where i' in [0,n), j' in [0,m)
    # U has shape (n*m, rank) = (10, 10)
    # U[i*m+j, i'*m+j'] = delta(i,i') * delta(j,j')
    # V has shape (m*p, rank) = (30, 10)
    # V[j*p+k, i'*m+j'] = delta(j,j') * a(k)
    # W has shape (p*n, rank) = (12, 10)
    # W[k*n+i, i'*m+j'] = delta(i,i') * 1/a(k)
    
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = sum_{i',j'} delta(i,i')delta(j,j') * delta(j,j')a(k) * delta(i,i')/a(k)
    # = delta(i,i')delta(j,j')delta(j,j')delta(i,i') * a(k)/a(k)
    # = 1 (when i'=i and j'=j, which is the only nonzero term)
    
    # Great! So rank = n*m = 10 is achievable!
    
    # Let's construct this:
    rank = n * m  # 10
    a = np.ones(p)  # a(k) = 1 for all k, simplest choice
    
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    
    for ip in range(n):
        for jp in range(m):
            c = ip * m + jp
            U[ip * m + jp, c] = 1.0
            for k in range(p):
                V[jp * p + k, c] = a[k]
            for k in range(p):
                W[k * n + ip, c] = 1.0 / a[k]
    
    # Verify
    T = np.ones((n, m, p))
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                T_recon[i, j, :] += uc[i, j] * vc[j, :] * wc[:, i]
    
    if np.max(np.abs(T_recon - T)) < 1e-10:
        # Great, rank 10 works!
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
