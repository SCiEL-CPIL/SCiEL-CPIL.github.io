"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _build_tensor(n, m, p):
    """Build the multiplication tensor as a 3D array T[u_idx, v_idx, w_idx]."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] equals the MM tensor."""
    T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(len(U[0])):
        u_col = np.array([U[i][r] for i in range(n*m)], dtype=np.float64)
        v_col = np.array([V[i][r] for i in range(m*p)], dtype=np.float64)
        w_col = np.array([W[i][r] for i in range(p*n)], dtype=np.float64)
        T_recon += np.einsum('a,b,c->abc', u_col, v_col, w_col)
    
    T_true = _build_tensor(n, m, p).astype(np.float64)
    return np.allclose(T_recon, T_true, atol=tol)

def _try_rank(n, m, p, rank, max_iters=5000, seed=42, deadline=None):
    """Try to find a rank-`rank` decomposition with integer entries in {-1,0,1}."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    best = None
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random integer entries in {-1, 0, 1}
        U = np.zeros((n*m, rank), dtype=np.int32)
        V = np.zeros((m*p, rank), dtype=np.int32)
        W = np.zeros((p*n, rank), dtype=np.int32)
        
        # Fill with random -1, 0, 1
        U = np_rng.integers(-1, 2, size=(n*m, rank))
        V = np_rng.integers(-1, 2, size=(m*p, rank))
        W = np_rng.integers(-1, 2, size=(p*n, rank))
        
        # Quick check: compute T_recon and compare
        T_recon = np.zeros((n*m, m*p, p*n), dtype=np.int32)
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        T_true = _build_tensor(n, m, p)
        if np.array_equal(T_recon, T_true):
            # Convert to list format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _strassen_style_3x3(n, m, p):
    """
    Try to use Strassen-like ideas. For 3x3x3, min rank is 23.
    For 3x3x8, we can potentially use block structure.
    Split B into two 3x4 blocks: B = [B1 | B2]
    Then C = [A*B1 | A*B2]
    If we have a rank-R decomposition for 3x3x3, we can use it for each block.
    But that gives rank 2*R. For R=23, that's 46 < 72.
    However, we need exact decomposition for the full tensor.
    """
    # Actually, let's try a different approach: 
    # The tensor for 3x3x8 has 3*3*8 = 72 nonzero entries.
    # Each rank-1 term can cover multiple entries if the pattern matches.
    # A rank-1 term U[:,r] ⊗ V[:,r] ⊗ W[:,r] has support on 
    # {i: U[i,r]≠0} × {j: V[j,r]≠0} × {k: W[k,r]≠0}
    # For it to be part of a decomposition of the MM tensor, the support must be
    # such that the sum over r gives exactly 1 at the right positions.
    
    # Let's try a structured approach: use the fact that m=3 is small.
    # For each column k of B (k=0..7), we compute A * B[:,k].
    # This is a 3x3 matrix-vector product, which has rank 23 (Strassen).
    # But we need a single decomposition for all 8 columns simultaneously.
    
    # Alternative: try rank 40-70 with random search
    return None

def construct():
    """
    Construct a tensor decomposition for matrix multiplication M_{n,m,p}.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + 130  # 130s deadline, leave 20s margin
    
    # First, try to find a better decomposition via random search
    # Start from lower ranks and go up
    
    # For 3x3x8, known bounds: schoolbook is 72.
    # Let's try ranks from 40 to 71
    best_result = None
    
    # Try a few random searches at various ranks
    for rank in range(40, min(72, schoolbook_rank)):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, max_iters=200, seed=rank*1000, deadline=deadline)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col_idx] = 1.0
                V[v_idx][col_idx] = 1.0
                W[w_idx][col_idx] = 1.0
                col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Simple self-check logic (not called by evaluator)
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    # Basic shape check
    assert len(res["U"]) == n * m
    assert len(res["V"]) == m * p
    assert len(res["W"]) == p * n
    rank = len(res["U"][0]) if n * m > 0 else 0
    assert len(res["V"][0]) == rank
    assert len(res["W"][0]) == rank
    
    # Verification on small random example (commented out to ensure speed, but good for sanity)
    # import random
    # N = 10
    # errors = 0
    # for _ in range(N):
    #     A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
    #     B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
    #     C_ref = [[sum(A[i][k] * B[k][l] for k in range(m)) for l in range(p)] for i in range(n)]
    #     C_tens = [[0 for _ in range(p)] for _ in range(n)]
    #     for r in range(rank):
    #         u_col = [res["U"][idx][r] for idx in range(n*m)]
    #         v_col = [res["V"][idx][r] for idx in range(m*p)]
    #         w_col = [res["W"][idx][r] for idx in range(p*n)]
    #         # Reconstruct A_part, B_part, C_part? 
    #         # U represents coefficients for A elements.
    #         # The tensor formula is T(i,j,k) = sum_r U(i,j,r) V(j,k,r) W(k,i,r)?
    #         # No, standard decomposition:
    #         # C_{i,l} = sum_{j} A_{i,j} B_{j,l}
    #         # Tensor: M_{i,j,l} = delta_{j,k}?? No.
    #         # The tensor is T_{i, j, l} where T_{i,j,l} = 1 if j==k else 0? No.
    #         # The multiplication tensor T has indices (i,j,l) for C_{i,l} += A_{i,j}B_{j,l}.
    #         # Wait, the interface says W[k*n+i, r]. This suggests W is associated with (k,i) i.e. (l,i) in (C_{l,i})?
    #         # Standard convention: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #         # Indices: i (0..n-1), j (0..m-1), k (0..p-1).
    #         # U index: i*m + j.
    #         # V index: j*p + k.
    #         # W index: k*n + i.
    #         # This matches C_{i,k} getting a contribution from A_{i,j} * B_{j,k}.
    #         # So C_calc[i][k] += sum_r U[i*m+j, r] * A[i,j] ... this is wrong.
    #         # The decomposition is: sum_{j} A_{i,j} B_{j,k} = sum_r (sum_j A_{i,j} U_{i,j,r}) (sum_k B_{j,k} V_{j,k,r})? No.
    #         # It is: sum_r ( sum_{j} A_{i,j} U_{i*m+j, r} ) * ( sum_{j'} B_{j', k} ... )
    #         # Actually, U_{i,j,r} is a coefficient.
    #         # The formula is: A_{i,j} B_{j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ??? No.
    #         # Tensor decomposition of MM(n,m,p):
    #         # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} is incorrect dimensionally if U,V,W are just vectors.
    #         # Correct interpretation:
    #         # The tensor T has entries T_{i,j,k}.
    #         # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} implies rank decomposition of the tensor.
    #         # For MM, T_{i,j,k} is 1 if j==k? No.
    #         # T_{i,j,k} corresponds to the operation C_{i,k} += A_{i,j} B_{j,k}.
    #         # So T_{i,j,k} is 1 for all i,j,k? No.
    #         # The multiplication map is linear in A and B.
    #         # C = A B.
    #         # Entry C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #         # This is a 3-linear form? No, it's bilinear in A and B.
    #         # The "tensor" of matrix multiplication is usually defined such that
    #         # <T, A \otimes B> = C.
    #         # Or T is a 3-tensor where T_{i,j,k} = \delta_{j,k} is not right.
    #         # The rank of matrix multiplication M_{n,m,p} is the number of multiplications.
    #         # Schoolbook: sum over j of A_{i,j} * B_{j,k}.
    #         # This requires n*m*p multiplications.
    #         # Decomposition:
    #         # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    #         # = sum_{j} (1 * A_{i,j}) * (1 * B_{j,k}) * (1 * delta_{i,k}?? No)
    #         # The decomposition form:
    #         # C_{i,k} = sum_{r=1}^{R} (sum_{j} U_{i,j,r} A_{i,j}) (sum_{j} V_{j,k,r} B_{j,k}) W_{k,i,r}?
    #         # No, standard form:
    #         # C_{i,k} = sum_r \tilde{A}_{i,r} \tilde{B}_{r,k}
    #         # Where \tilde{A}_{i,r} = sum_j U_{i,j,r} A_{i,j}
    #         # \tilde{B}_{r,k} = sum_j V_{j,k,r} B_{j,k}
    #         # And there is no W if we output matrix product directly?
    #         # But the problem provides U, V, W.
    #         # Usually for tensor T_{i,j,k}:
    #         # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} is a decomposition of a 3-tensor.
    #         # Does MM fit this?
    #         # The "multiplication tensor" M_{n,m,p} has entries M_{i,j,k} = 1 if we are multiplying the j-th element of row i of A and k-th element of col? 
    #         # Actually, the tensor M_{n,m,p} is defined by M_{i,j,k} = \delta_{j,k} is not standard.
    #         # Standard: The tensor has indices (i, j, k) representing A_{i,j}, B_{j,k}, C_{i,k}.
    #         # The structure tensor is T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i} ?
    #         # No, the tensor is $ \sum_{i,j,k} a_{i,j} b_{j,k} \rightarrow c_{i,k} $.
    #         # As a linear map $ V_A \otimes V_B \to V_C $.
    #         # Rank is dim( smallest W s.t. $ V_A \otimes V_B \to W \to V_C $ is an isomorphism onto image?).
    #         
    #         Let's verify the schoolbook against the formula:
    #         For each (i,j,k), we have a rank 1 term.
    #         U[i*m+j, r] = 1.
    #         V[j*p+k, r] = 1.
    #         W[k*n+i, r] = 1.
    #         Does this reconstruct $ A B $?
    #         The term contributes to C?
    #         Usually, the reconstruction is:
    #         $ C_{i,k} = \sum_r \tilde{A}_{i,r} \tilde{B}_{r,k} $ ??
    #         If so, where does W go?
    #         If W is provided, maybe the form is:
    #         $ T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} $
    #         And $ T $ is the multiplication tensor?
    #         What is $ T_{i,j,k} $ for MM?
    #         It is 1 if the indices match the multiplication pattern?
    #         For MM, $ T_{i,j,k} $ is not sparse in a way that $ \delta_{jk} $ works directly in all 3 indices unless we view it differently.
    #         However, the schoolbook decomposition IS the standard rank $nmp$ decomposition.
    #         If the indices are set as in the seed:
    #         r maps to (i,j,k).
    #         U[i*m+j, r] = 1.
    #         V[j*p+k, r] = 1.
    #         W[k*n+i, r] = 1.
    #         
    #         Let's check the sum:
    #         $\sum_r U_{i,j,r} V_{j',k,r} W_{k',i,r}$?
    #         If we fix input indices for A (i,j) and B (j',k') and output (i',k'):
    #         This is getting confusing without the exact evaluator spec.
    #         However, the seed code works (score 1.0).
    #         So I will trust the seed's index mapping.

    # Time check
    elapsed = time.time() - start_time
    print(f"Constructed in {elapsed:.4f}s")
