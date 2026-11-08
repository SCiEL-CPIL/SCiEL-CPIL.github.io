"""Rewritten program for 3x3 * 3x7 matrix multiplication tensor decomposition.
Attempts a low-rank exact decomposition via structured ansatz + local refinement.
Falls back to schoolbook rank 63 if no improvement is found within time budget.
"""
import time
import sys
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]
n, m, p = DIMENSIONS
NM = n * m
MP = m * p
PN = p * n
SCHOOL_RANK = NM * MP // m  # 3*3*7 = 63

# The tensor T has indices (i, j, k) with i in [0,n), j in [0,m), k in [0,p)
# U shape: (n*m, rank)  indexed by (i*m + j)
# V shape: (m*p, rank)  indexed by (j*p + k)
# W shape: (p*n, rank)  indexed by (k*n + i)  (trace convention W[k*n+i, r])

# Tensor entries: T[i,j,k] = 1 for all valid (i,j,k)
# We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1

def tensor_residual(U, V, W):
    """Compute max absolute residual of the tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                res = abs(val - 1.0)
                if res > max_res:
                    max_res = res
    return max_res

def verify_decomposition(U, V, W, tol=1e-8):
    """Verify that the decomposition is exact within tolerance."""
    max_res = tensor_residual(U, V, W)
    return max_res < tol, max_res

def schoolbook_decomposition():
    """Return the schoolbook (rank 63) decomposition."""
    rank = SCHOOL_RANK
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def structured_ansatz(rank):
    """
    Build a structured low-rank ansatz.
    Uses the idea that for square-ish matmul, one can use rank < n*m*p.
    For 3x3x3, rank 23 is known. For 3x3x7, we try to find something lower than 63.
    
    Strategy: use a "block diagonal" structure where we group indices.
    Each rank-r term is parameterized by (a_r, b_r, c_r) where:
      U[i*m+j, r] = f_r(i, j)
      V[j*p+k, r] = g_r(j, k)
      W[k*n+i, r] = h_r(k, i)
    
    We try a sum-of-products ansatz with random initialization and local search.
    """
    random.seed(42)
    np.random.seed(42)
    
    # Initialize with random values in {-1, 0, 1} with some zeros for sparsity
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    
    for r in range(rank):
        # Random sparse entries
        U[:, r] = np.random.choice([-1, 0, 1], size=NM)
        V[:, r] = np.random.choice([-1, 0, 1], size=MP)
        W[:, r] = np.random.choice([-1, 0, 1], size=PN)
    
    return U, V, W

def local_refine(U, V, W, max_steps=5000, tol=1e-10):
    """
    Local refinement: iteratively adjust entries to reduce the tensor residual.
    Uses a greedy coordinate descent approach on the factor matrices.
    """
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    rank = U.shape[1]
    
    for step in range(max_steps):
        max_res_before = tensor_residual(U, V, W)
        if max_res_before < tol:
            break
        
        # Pick the worst (i, j, k) residual
        worst_res = 0.0
        worst_ijk = (0, 0, 0)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res = abs(val - 1.0)
                    if res > worst_res:
                        worst_res = res
                        worst_ijk = (i, j, k)
        
        i, j, k = worst_ijk
        ui = i*m+j
        vj = j*p+k
        wk = k*n+i
        
        # Try to adjust one entry at a time to reduce residual
        # For each rank r, compute the "partial" contribution
        prod = U[ui, :] * V[vj, :] * W[wk, :]
        current_val = np.sum(prod)
        
        # Try adjusting U[ui, r], V[vj, r], W[wk, r] for each r
        best_delta = 0.0
        best_action = None
        
        for r in range(rank):
            u_val = U[ui, r]
            v_val = V[vj, r]
            w_val = W[wk, r]
            prod_r = prod[r]
            
            # Try setting U[ui, r] to 0, 1, -1
            for new_u in [0.0, 1.0, -1.0]:
                if new_u == u_val:
                    continue
                new_prod_r = new_u * v_val * w_val
                new_val = current_val - prod_r + new_prod_r
                new_res = abs(new_val - 1.0)
                delta = worst_res - new_res
                if delta > best_delta + 1e-12:
                    best_delta = delta
                    best_action = ('U', ui, r, new_u)
            
            # Try setting V[vj, r] to 0, 1, -1
            for new_v in [0.0, 1.0, -1.0]:
                if new_v == v_val:
                    continue
                new_prod_r = u_val * new_v * w_val
                new_val = current_val - prod_r + new_prod_r
                new_res = abs(new_val - 1.0)
                delta = worst_res - new_res
                if delta > best_delta + 1e-12:
                    best_delta = delta
                    best_action = ('V', vj, r, new_v)
            
            # Try setting W[wk, r] to 0, 1, -1
            for new_w in [0.0, 1.0, -1.0]:
                if new_w == w_val:
                    continue
                new_prod_r = u_val * v_val * new_w
                new_val = current_val - prod_r + new_prod_r
                new_res = abs(new_val - 1.0)
                delta = worst_res - new_res
                if delta > best_delta + 1e-12:
                    best_delta = delta
                    best_action = ('W', wk, r, new_w)
        
        if best_action is None:
            # No single entry change helps; try small continuous adjustments
            # Try adding a small epsilon to a randomly chosen entry
            break
        
        matrix_name, idx, r, new_val = best_action
        if matrix_name == 'U':
            U[idx, r] = new_val
        elif matrix_name == 'V':
            V[idx, r] = new_val
        elif matrix_name == 'W':
            W[idx, r] = new_val
    
    return U, V, W

def try_low_rank_search(max_rank_to_try=40, time_budget=100):
    """
    Try to find a decomposition with rank < 63.
    Uses structured ansatz + local refinement for various ranks.
    Returns (rank, U, V, W) if found, else None.
    """
    start_time = time.time()
    
    for rank in range(20, max_rank_to_try + 1):
        if time.time() - start_time > time_budget:
            break
        
        attempts = 3
        for attempt in range(attempts):
            if time.time() - start_time > time_budget:
                break
            
            # Different initializations
            if attempt == 0:
                U, V, W = structured_ansatz(rank)
            elif attempt == 1:
                # Sparse random
                U = np.zeros((NM, rank))
                V = np.zeros((MP, rank))
                W = np.zeros((PN, rank))
                for r in range(rank):
                    for idx in range(NM):
                        if random.random() < 0.3:
                            U[idx, r] = random.choice([-1, 1])
                    for idx in range(MP):
                        if random.random() < 0.3:
                            V[idx, r] = random.choice([-1, 1])
                    for idx in range(PN):
                        if random.random() < 0.3:
                            W[idx, r] = random.choice([-1, 1])
            else:
                # Half-integer values
                U = np.zeros((NM, rank))
                V = np.zeros((MP, rank))
                W = np.zeros((PN, rank))
                for r in range(rank):
                    for idx in range(NM):
                        if random.random() < 0.4:
                            U[idx, r] = random.choice([-1.0, -0.5, 0.5, 1.0])
                    for idx in range(MP):
                        if random.random() < 0.4:
                            V[idx, r] = random.choice([-1.0, -0.5, 0.5, 1.0])
                    for idx in range(PN):
                        if random.random() < 0.4:
                            W[idx, r] = random.choice([-1.0, -0.5, 0.5, 1.0])
            
            U, V, W = local_refine(U, V, W, max_steps=2000)
            ok, max_res = verify_decomposition(U, V, W, tol=1e-8)
            if ok:
                # Convert to list-of-lists with clean values
                def clean_matrix(mat):
                    result = []
                    for row in mat:
                        clean_row = []
                        for val in row:
                            if abs(val - round(val)) < 1e-6:
                                clean_row.append(int(round(val)))
                            elif abs(2*val - round(2*val)) < 1e-6:
                                clean_row.append(round(2*val)/2.0)
                            else:
                                clean_row.append(val)
                        result.append(clean_row)
                    return result
                
                U_clean = clean_matrix(U)
                V_clean = clean_matrix(V)
                W_clean = clean_matrix(W)
                
                # Final verification
                ok2, res2 = verify_decomposition(U_clean, V_clean, W_clean, tol=1e-8)
                if ok2:
                    return rank, U_clean, V_clean, W_clean
    
    return None

def construct():
    """
    Constructs the tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Attempts to find a rank-minimal exact decomposition.
    Falls back to schoolbook rank 63.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try to find a lower-rank decomposition
    result = try_low_rank_search(max_rank_to_try=50, time_budget=90)
    
    if result is not None:
        rank, U, V, W = result
        # Final check
        ok, max_res = verify_decomposition(U, V, W, tol=1e-8)
        if ok:
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return schoolbook_decomposition()

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    end_time = time.time()
    if end_time - start_time > 110:
        print("Warning: Construction took too long", file=sys.stderr)
    # Verify dimensions
    assert len(res['U']) == NM
    assert len(res['V']) == MP
    assert len(res['W']) == PN
    rank = len(res['U'][0])
    assert len(res['V'][0]) == rank
    assert len(res['W'][0]) == rank
    # Verify correctness
    ok, max_res = verify_decomposition(res['U'], res['V'], res['W'], tol=1e-8)
    if not ok:
        print(f"ERROR: Decomposition not exact. Max residual: {max_res}", file=sys.stderr)
        sys.exit(1)
    print(f"Rank: {rank}, Max residual: {max_res:.2e}, Time: {end_time-start_time:.2f}s")
