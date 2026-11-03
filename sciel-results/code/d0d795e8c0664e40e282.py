"""Program for matrix multiplication 3x3 * 3x5 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
n, m, p = DIMENSIONS
N = n  # 3
M = m  # 3
P = p  # 5
NM = N * M   # 9
MP = M * P   # 15
PN = P * N   # 15
SCHOOLBOOK_RANK = NM * MP // M * N  # This is n*m*p = 45
# Actually schoolbook rank = n*m*p = 3*3*5 = 45

def _build_schoolbook():
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = N * M * P
    U = [[0.0] * rank for _ in range(NM)]
    V = [[0.0] * rank for _ in range(MP)]
    W = [[0.0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1.0
                V[j * P + k][col] = 1.0
                W[k * N + i][col] = 1.0
                col += 1
    return U, V, W, rank

def _verify(U, V, W, tol=1e-10):
    """Verify that sum_r U[i,j,r] * V[j,k,r] == W[k,i,r] for all (i,j,k).
    Convention: W[k*n+i, r] so W index is k*N + i.
    We need: for all (i, j, k), sum_r U[i*M+j, r] * V[j*P+k, r] == 1.0
    And the W factor: W[k*N+i, r] should be 1.0 for the matching term.
    
    Actually the tensor decomposition is:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    For matrix multiplication, T[i,j,k] = 1 for all (i,j,k) (since (AB)[i,k] = sum_j A[i,j]*B[j,k]).
    
    So we need: sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] == 1 for all i,j,k.
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Check each (i, j, k)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for r in range(rank):
                    s += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                if abs(s - 1.0) > tol:
                    return False, s, (i, j, k)
    return True, 0.0, None

def _try_random_search(deadline, best_result):
    """Try to find a lower-rank decomposition using random search with local refinement."""
    best_U, best_V, best_W, best_rank = best_result
    best_err = float('inf')
    
    # Try ranks from a lower bound up to schoolbook - 1
    # Theoretical lower bound: max(N*M, M*P, P*N) = max(9, 15, 15) = 15
    # But for matrix multiplication, Strassen-type bounds suggest higher.
    # For 3x3x5, let's try from 30 up to 44
    
    for target_rank in range(30, min(SCHOOLBOOK_RANK, 45)):
        if time.time() > deadline:
            break
        max_iters = 20000
        for restart in range(5):
            if time.time() > deadline:
                break
            U = np.random.uniform(-1, 1, (NM, target_rank))
            V = np.random.uniform(-1, 1, (MP, target_rank))
            W = np.random.uniform(-1, 1, (PN, target_rank))
            
            # Project to half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                # Compute error
                # E = sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] - 1
                # Use numpy for efficiency
                # Build full tensor product sum
                # For each r, outer product U[:,r] * V[:,r] * W[:,r] contributes to T
                # T[i,j,k] = sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
                
                # Efficient computation:
                # For each r, create a 3D array: U_col reshaped to (N,M), V_col to (M,P), W_col to (P,N)
                # contribution[i,j,k] = U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
                # This is a tensor product: u_r outer v_r outer w_r where indices are permuted
                
                # Let's compute the full tensor
                T_comp = np.zeros((N, M, P))
                for r in range(target_rank):
                    u = U[:, r].reshape(N, M)
                    v = V[:, r].reshape(M, P)
                    w = W[:, r].reshape(P, N)
                    # T_comp[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                    # This is tricky with standard einsum due to the i-index in w
                    # Let's just do it explicitly for small dimensions
                    for i in range(N):
                        for j in range(M):
                            for k in range(P):
                                T_comp[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                
                err = np.abs(T_comp - 1.0).max()
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    best_rank = target_rank
                
                if err < 1e-10:
                    # Found exact solution!
                    return _convert_to_lists(best_U, best_V, best_W, best_rank)
                
                # Local search: perturb one random element
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    idx = random.randrange(NM)
                    U[idx, random.randrange(target_rank)] += random.choice([-0.5, 0.5])
                elif mat_choice == 'V':
                    idx = random.randrange(MP)
                    V[idx, random.randrange(target_rank)] += random.choice([-0.5, 0.5])
                else:
                    idx = random.randrange(PN)
                    W[idx, random.randrange(target_rank)] += random.choice([-0.5, 0.5])
                
                # Snap to half-integers
                U = np.clip(np.round(U * 2) / 2, -3, 3)
                V = np.clip(np.round(V * 2) / 2, -3, 3)
                W = np.clip(np.round(W * 2) / 2, -3, 3)
    
    return _convert_to_lists(best_U, best_V, best_W, best_rank)

def _convert_to_lists(U, V, W, rank):
    """Convert numpy arrays to lists of lists with half-integer entries."""
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(W.shape[0])]
    return U_list, V_list, W_list, rank

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook()
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    
    # Try to find better
    try:
        U_try, V_try, W_try, rank_try = _try_random_search(deadline, best_result)
        # Verify
        valid, err, loc = _verify(U_try, V_try, W_try)
        if valid and rank_try < rank_sb:
            return {"U": U_try, "V": V_try, "W": W_try}
    except Exception:
        pass
    
    # Return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
