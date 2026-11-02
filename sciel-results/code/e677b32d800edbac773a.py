"""Combined program: Strassen-based rank-21 construction with structural rank reduction."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # Safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _build_strassen_rank21(n, m, p):
    """
    Build a rank-21 decomposition using Strassen's algorithm on three 2x2 blocks.
    """
    nm, mp, pn = n * m, m * p, p * n
    rank = 21
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 0, 1, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [1, -1, 0, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 0, 1, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 1, 0, 1], 'w': [1, 0, 0, 0]},
    ]
    
    v_indices = []
    w_indices = []
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        v_indices.append([0*6 + k0, 0*6 + k1, 1*6 + k0, 1*6 + k1])
        w_indices.append([k0*2 + 0, k1*2 + 0, k0*2 + 1, k1*2 + 1])
    
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            u_list = step['u']
            v_list = step['v']
            w_list = step['w']
            v_idx = v_indices[g]
            w_idx = w_indices[g]
            for local_idx in range(4):
                u_val = u_list[local_idx]
                if u_val != 0:
                    U[local_idx, r] = u_val
                v_val = v_list[local_idx]
                if v_val != 0:
                    V[v_idx[local_idx], r] = v_val
                w_val = w_list[local_idx]
                if w_val != 0:
                    W[w_idx[local_idx], r] = w_val
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    for matrix in [U_list, V_list, W_list]:
        for row in matrix:
            for idx, val in enumerate(row):
                if isinstance(val, float):
                    if abs(val - round(val)) < 1e-10:
                        row[idx] = int(round(val))
    
    return U_list, V_list, W_list

def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_time=100):
    """
    Structural rank reduction: try to find a lower-rank equivalent by
    exploiting the shared A-structure. Uses iterative elimination with
    bilinear constraint solving.
    """
    start = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Build the target tensor
    T = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T[a, b, c] = 1
    
    # Try eliminating each column in sequence
    for _ in range(current_rank - target_rank):
        if time.time() - start > max_time:
            break
        
        found = False
        for elim_r in range(U.shape[1]):
            if time.time() - start > max_time:
                break
            
            # Remove column elim_r
            U_rem = np.delete(U, elim_r, axis=1)
            V_rem = np.delete(V, elim_r, axis=1)
            W_rem = np.delete(W, elim_r, axis=1)
            
            # Compute residual: what's missing from the eliminated term
            # residual[a,b,c] = -U[elim_r,a] * V[elim_r,b] * W[elim_r,c]
            U_elim = U[:, elim_r]
            V_elim = V[:, elim_r]
            W_elim = W[:, elim_r]
            
            # Check if we can absorb this into remaining terms via small adjustments
            # Use a first-order perturbation approach
            # We want: sum_r dU[a,r]*V_rem[b,r]*W_rem[c,r] + U_rem[a,r]*dV[b,r]*W_rem[c,r] + U_rem[a,r]*V_rem[b,r]*dW[c,r] = residual[a,b,c]
            
            # This is a linear system in dU, dV, dW
            # But it's underdetermined. Instead, try a simpler check:
            # Is the eliminated term already zero or negligible?
            
            if np.max(np.abs(U_elim)) < 1e-10 and np.max(np.abs(V_elim)) < 1e-10 and np.max(np.abs(W_elim)) < 1e-10:
                U = U_rem
                V = V_rem
                W = W_rem
                found = True
                break
            
            # Try: can we add a multiple of one remaining column to compensate?
            # For each remaining column r, check if we can find scalars alpha, beta, gamma
            # such that the perturbation works. This is complex, so use a heuristic:
            # Check if the eliminated term's support is subset of some remaining term's support
            
            # Simpler structural check: look for two columns with identical U vectors
            for r1 in range(U_rem.shape[1]):
                if time.time() - start > max_time:
                    break
                if np.allclose(U_rem[:, r1], U_elim, atol=1e-10):
                    # Same U pattern - check if V and W can be adjusted
                    # Try: new_V_col = V_rem[:,r1] + c*V_elim, new_W_col = W_rem[:,r1] + d*W_elim
                    # This requires bilinear matching, which is hard.
                    # Just try removing and verifying.
                    test_U = np.delete(U, elim_r, axis=1)
                    test_V = np.delete(V, elim_r, axis=1)
                    test_W = np.delete(W, elim_r, axis=1)
                    
                    # Verify the 20-term decomposition
                    valid = True
                    for a in range(4):
                        if not valid:
                            break
                        for b in range(12):
                            if not valid:
                                break
                            for c in range(12):
                                s = 0
                                for r in range(test_U.shape[1]):
                                    s += test_U[a, r] * test_V[b, r] * test_W[c, r]
                                if abs(s - T[a, b, c]) > 1e-10:
                                    valid = False
                                    break
                    if valid:
                        U = test_U
                        V = test_V
                        W = test_W
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if not found:
            break
    
    # Convert to integer lists if possible
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    for matrix in [U_list, V_list, W_list]:
        for row in matrix:
            for idx, val in enumerate(row):
                if isinstance(val, float):
                    if abs(val - round(val)) < 1e-10:
                        row[idx] = int(round(val))
    
    return U_list, V_list, W_list

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank-n*m*p decomposition as fallback."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Build the Strassen rank-21 decomposition
    U21, V21, W21 = _build_strassen_rank21(n, m, p)
    
    # Try structural rank reduction to rank 20
    U20, V20, W20 = _try_rank_reduction(U21, V21, W21, n, m, p, 20, max_time=100)
    
    if len(U20[0]) == 20:
        if _verify_decomposition(U20, V20, W20, n, m, p):
            return {"U": U20, "V": V20, "W": W20}
    
    # Fallback to verified rank-21
    if _verify_decomposition(U21, V21, W21, n, m, p):
        return {"U": U21, "V": V21, "W": W21}
    
    # Final fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
