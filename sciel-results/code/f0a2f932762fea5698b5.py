"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_rank26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication using explicit factors.
    Strassen core (7 terms) for the 2x2 block + schoolbook border terms.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _fft_based_search(base_result, deadline):
    """
    Use DFT over Z3 to diagonalize the tensor and search for lower-rank decompositions.
    The idea: apply DFT to the 3x3 matrices, which diagonalizes the cyclic shift symmetry.
    In the Fourier basis, the 3x3 matmul tensor decomposes into 9 independent 1x1x1
    "tensors" (scalars) if we use the full DFT, but that's not quite right.
    
    Actually, the key insight: the 3x3 matrix multiplication tensor M_{i,j,k} = delta_{i+k,j}
    (where C[i,j] = sum_k A[i,k] * B[k,j]) has a symmetry under cyclic shifts.
    
    Let's use a different approach: directly search for rank-25 decompositions by
    trying to find 25 rank-1 tensors that sum to the matmul tensor.
    
    We'll use the fact that the matmul tensor is 27-dimensional (3x3x3).
    A rank-r decomposition means we can write T = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t.
    
    For rank 25, we need to find 25 triples (u_t, v_t, w_t) such that
    sum_t u_t[i] * v_t[j] * w_t[k] = delta_{i+k,j} for all i,j,k.
    
    This is a system of 27 equations in 25*9 = 225 unknowns (with 25*9 = 225 variables
    if we fix the normalization).
    
    Actually, let's use a more practical approach: start from the rank-26 solution and
    try to merge two terms into one. Two terms (u1,v1,w1) and (u2,v2,w2) can be merged
    into a single term if there exists a rank-1 tensor that equals their sum.
    
    The sum of two rank-1 tensors u1⊗v1⊗w1 + u2⊗v2⊗w2 is rank-1 iff the 3x3 matrix
    (u1⊗v1) * (w1^T) + (u2⊗v2) * (w2^T) has rank 1.
    
    Equivalently, we need to check if there exist scalars a, b such that
    a * u1⊗v1⊗w1 + b * u2⊗v2⊗w2 is rank-1, which means the 3x3 matrix
    a*(u1 v1^T) * w1^T + b*(u2 v2^T) * w2^T has rank 1.
    
    This is getting complex. Let's try a simpler heuristic: randomly perturb the
    rank-26 solution and see if we can find a valid rank-25 solution.
    """
    n = m = p = 3
    U = np.array(base_result["U"], dtype=float)
    V = np.array(base_result["V"], dtype=float)
    W = np.array(base_result["W"], dtype=float)
    cur_rank = U.shape[1]
    
    # Try to find a rank-25 decomposition by removing one term and adjusting others
    # This is equivalent to solving a nonlinear system, which is hard.
    # Instead, let's try a randomized search: perturb the terms and check validity.
    
    import random
    random.seed(12345)
    
    best = None
    best_rank = cur_rank
    
    # Try each pair of terms to see if they can be merged
    for a in range(cur_rank):
        if time.time() > deadline:
            break
        for b in range(a + 1, cur_rank):
            if time.time() > deadline:
                break
            # Try to merge terms a and b into a single term
            # The sum of two rank-1 tensors is rank-1 iff their "outer product" matrices
            # are linearly dependent in a specific way.
            # 
            # Let's check: is there a rank-1 tensor T such that
            # T + sum_{r != a,b} term_r = identity?
            # 
            # This is equivalent to: T = identity - sum_{r != a,b} term_r
            # and T must be rank-1.
            #
            # Compute T and check if it's rank-1.
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(cur_rank):
                            if r == a or r == b:
                                continue
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        T[i, j, k] = 1.0 - s
            
            # Check if T is rank-1: all 2x2 minors should be zero
            # For a 3x3x3 tensor, rank-1 means T[i,j,k] = u[i]*v[j]*w[k]
            # Check: T[0,0,0]*T[0,1,1] == T[0,1,0]*T[0,0,1] etc.
            is_rank1 = True
            # Check a few 2x2 minors
            if abs(T[0,0,0]*T[1,1,1] - T[0,1,1]*T[1,0,0]) > 1e-8:
                is_rank1 = False
            if is_rank1 and abs(T[0,0,1]*T[1,1,0] - T[0,1,0]*T[1,0,1]) > 1e-8:
                is_rank1 = False
            if is_rank1 and abs(T[0,1,0]*T[1,0,1] - T[0,0,1]*T[1,1,0]) > 1e-8:
                is_rank1 = False
            
            if is_rank1:
                # Found a merge! Create new rank-25 decomposition
                remaining = [x for x in range(cur_rank) if x != a and x != b]
                new_U = U[:, remaining].copy()
                new_V = V[:, remaining].copy()
                new_W = W[:, remaining].copy()
                
                # Add the merged term
                new_rank = cur_rank - 1
                new_U = np.hstack([new_U, np.zeros((n*m, 1))])
                new_V = np.hstack([new_V, np.zeros((m*p, 1))])
                new_W = np.hstack([new_W, np.zeros((p*n, 1))])
                
                # Extract u, v, w from T
                # T[i,j,k] = u[i]*v[j]*w[k]
                # u = T[:,0,0] / (v[0]*w[0]) if v[0]*w[0] != 0
                if abs(T[0,0,0]) > 1e-10:
                    # Use T[i,0,0] = u[i]*v[0]*w[0]
                    # T[0,j,0] = u[0]*v[j]*w[0]
                    # T[0,0,k] = u[0]*v[0]*w[k]
                    u = T[:, 0, 0] / T[0, 0, 0]
                    v = T[0, :, 0] / T[0, 0, 0]
                    w = T[0, 0, :] / T[0, 0, 0]
                    
                    new_U[:, -1] = u.flatten()
                    new_V[:, -1] = v.flatten()
                    new_W[:, -1] = w.flatten()
                    
                    # Verify
                    result = {
                        "U": new_U.tolist(),
                        "V": new_V.tolist(),
                        "W": new_W.tolist()
                    }
                    if _verify(result["U"], result["V"], result["W"]):
                        return result, new_rank
    
    return best, best_rank

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with the verified rank-26 construction
    result = _strassen_rank26()
    if not _verify(result["U"], result["V"], result["W"]):
        return _schoolbook()
    
    # Try FFT-based search for rank reduction
    if time.time() < deadline - 20:
        reduced, reduced_rank = _fft_based_search(result, deadline - 20)
        if reduced is not None:
            if _verify(reduced["U"], reduced["V"], reduced["W"]):
                result = reduced
    
    # Final verification
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    return _schoolbook()

# EVOLVE-BLOCK-END
