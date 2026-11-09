"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify(decomp, n, m, p):
    """Verify that the tensor decomposition exactly multiplies the matrices."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    r = U.shape[1]
    
    # Build index arrays for vectorized check
    # For each (i,j,k): check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_exact(decomp, n, m, p):
    """Exact verification using fractions."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    r = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                for idx in range(r):
                    val += Fraction(U[i*m + j][idx]) * Fraction(V[j*p + k][idx]) * Fraction(W[k*n + i][idx])
                if val != 1:
                    return False
    return True

def _snap_to_half_integers(U, V, W, n, m, p):
    """Try to snap to half-integer values and verify."""
    for scale in [2, 4, 8, 16]:
        U_snap = np.round(U * scale) / scale
        V_snap = np.round(V * scale) / scale
        W_snap = np.round(W * scale) / scale
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        decomp = {"U": U_list, "V": V_list, "W": W_list}
        if _verify(decomp, n, m, p):
            return decomp
    
    # Also try original
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    decomp = {"U": U_list, "V": V_list, "W": W_list}
    if _verify(decomp, n, m, p):
        return decomp
    
    return None

def _try_rank_vectorized(n, m, p, rank, seed=42, time_limit=10.0):
    """Try to find a rank-r decomposition using vectorized Adam optimization."""
    rng = np.random.RandomState(seed)
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    start_time = time.time()
    
    # Precompute index mappings for vectorized gradient computation
    # For each (i,j,k) triple, we need u_idx, v_idx, w_idx
    u_indices = np.array([[i * m + j for j in range(m)] for i in range(n)])  # shape (n, m)
    v_indices = np.array([[j * p + k for k in range(p)] for j in range(m)])  # shape (m, p)
    w_indices = np.array([[k * n + i for i in range(n)] for k in range(p)])  # shape (p, n)
    
    # Flatten the triple (i,j,k) -> (u_idx, v_idx, w_idx)
    all_u = []
    all_v = []
    all_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_u.append(i * m + j)
                all_v.append(j * p + k)
                all_w.append(k * n + i)
    all_u = np.array(all_u)  # length n*m*p
    all_v = np.array(all_v)
    all_w = np.array(all_w)
    
    num_triples = len(all_u)
    
    # Initialize
    U = np.zeros((num_u, rank))
    V = np.zeros((num_v, rank))
    W = np.zeros((num_w, rank))
    
    # Initialize with schoolbook pattern for first min(rank, n*m*p) columns
    schoolbook_rank = min(rank, n * m * p)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if column < rank:
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
    
    # Random perturbation
    U += rng.randn(num_u, rank) * 0.05
    V += rng.randn(num_v, rank) * 0.05
    W += rng.randn(num_w, rank) * 0.05
    
    # Adam optimizer state
    lr = 0.02
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    patience = 300
    no_improve = 0
    t = 0
    
    for it in range(50000):
        if time.time() - start_time > time_limit:
            break
        
        t += 1
        
        # Vectorized gradient computation
        # For each triple t: val_t = sum_r U[u_t, r] * V[v_t, r] * W[w_t, r]
        # We can compute all vals at once:
        # U_rows = U[all_u] -> shape (num_triples, rank)
        # V_rows = V[all_v] -> shape (num_triples, rank)
        # W_rows = W[all_w] -> shape (num_triples, rank)
        # vals = sum_r U_rows * V_rows * W_rows -> shape (num_triples,)
        
        U_rows = U[all_u]  # (num_triples, rank)
        V_rows = V[all_v]  # (num_triples, rank)
        W_rows = W[all_w]  # (num_triples, rank)
        
        vals = np.sum(U_rows * V_rows * W_rows, axis=1)  # (num_triples,)
        diffs = vals - 1.0  # (num_triples,)
        loss = np.sum(diffs * diffs)
        
        # Gradients:
        # grad_U[u_idx] = sum over triples t where u_t == u_idx of 2*diff_t * V[v_t] * W[w_t]
        # grad_V[v_idx] = sum over triples t where v_t == v_idx of 2*diff_t * U[u_t] * W[w_t]
        # grad_W[w_idx] = sum over triples t where w_t == w_idx of 2*diff_t * U[u_t] * V[v_t]
        
        # For U: contribution from triple t is 2*diff_t * V[v_t, r] * W[w_t, r] for each r
        # We need to scatter-add these contributions to grad_U[all_u[t], r]
        
        # Vectorized: 
        # contrib_U[t, r] = 2 * diff[t] * V_rows[t, r] * W_rows[t, r]
        # Then grad_U[u_idx] = sum over t where all_u[t] == u_idx of contrib_U[t, :]
        
        contrib_U = 2.0 * diffs[:, np.newaxis] * V_rows * W_rows  # (num_triples, rank)
        contrib_V = 2.0 * diffs[:, np.newaxis] * U_rows * W_rows  # (num_triples, rank)
        contrib_W = 2.0 * diffs[:, np.newaxis] * U_rows * V_rows  # (num_triples, rank)
        
        # Scatter add
        grad_U = np.zeros((num_u, rank))
        grad_V = np.zeros((num_v, rank))
        grad_W = np.zeros((num_w, rank))
        
        np.add.at(grad_U, all_u, contrib_U)
        np.add.at(grad_V, all_v, contrib_V)
        np.add.at(grad_W, all_w, contrib_W)
        
        # Adam update
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * (grad_U ** 2)
        m_hat_U = m_U / (1 - beta1 ** t)
        v_hat_U = v_U / (1 - beta2 ** t)
        U -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * (grad_V ** 2)
        m_hat_V = m_V / (1 - beta1 ** t)
        v_hat_V = v_V / (1 - beta2 ** t)
        V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * (grad_W ** 2)
        m_hat_W = m_W / (1 - beta1 ** t)
        v_hat_W = v_W / (1 - beta2 ** t)
        W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Adaptive learning rate
        if no_improve > patience:
            lr *= 0.5
            no_improve = 0
            if lr < 1e-10:
                break
        
        if loss < 1e-14:
            break
    
    # Use best found
    U, V, W = best_U, best_V, best_W
    
    if best_loss < 1e-10:
        # Try to snap to half-integers
        result = _snap_to_half_integers(U, V, W, n, m, p)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    best_decomp = _schoolbook()
    best_rank = best_decomp["rank"]
    
    # Try ranks from lower to upper
    for rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            
            remaining_time = deadline - time.time()
            if remaining_time < 10:
                break
            
            time_per_try = min(8.0, remaining_time / 2.0)
            result = _try_rank_vectorized(n, m, p, rank, seed=seed + 100, time_limit=time_per_try)
            if result is not None:
                if _verify(result, n, m, p):
                    best_decomp = result
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
# EVOLVE-BLOCK-END
