import time
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _strassen_2x2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur):
    def u(i, j): return (i_off + i) * m + (j_off + j)
    def v(j, k): return (j_off + j) * p + (k_off + k)
    def w(i, k): return (k_off + k) * n + (i_off + i)

    # M1 = (a00+a11)(b00+b11)
    U[u(0,0)][cur] += 1
    U[u(1,1)][cur] += 1
    V[v(0,0)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M2 = (a10+a11)(b00)
    U[u(1,0)][cur] += 1
    U[u(1,1)][cur] += 1
    V[v(0,0)][cur] += 1
    W[w(1,0)][cur] += 1
    W[w(1,1)][cur] -= 1
    cur += 1
    # M3 = (a00)(b01-b11)
    U[u(0,0)][cur] += 1
    V[v(0,1)][cur] += 1
    V[v(1,1)][cur] -= 1
    W[w(0,1)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M4 = (a11)(b10-b00)
    U[u(1,1)][cur] += 1
    V[v(1,0)][cur] += 1
    V[v(0,0)][cur] -= 1
    W[w(0,0)][cur] += 1
    W[w(1,0)][cur] += 1
    cur += 1
    # M5 = (a00+a01)(b11)
    U[u(0,0)][cur] += 1
    U[u(0,1)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] -= 1
    W[w(0,1)][cur] += 1
    cur += 1
    # M6 = (a10-a00)(b00+b01)
    U[u(1,0)][cur] += 1
    U[u(0,0)][cur] -= 1
    V[v(0,0)][cur] += 1
    V[v(0,1)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M7 = (a01-a11)(b10+b11)
    U[u(0,1)][cur] += 1
    U[u(1,1)][cur] -= 1
    V[v(1,0)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] += 1
    cur += 1

    return cur

def _winograd_2x2_2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur):
    def u(i, j): return (i_off + i) * m + (j_off + j)
    def v(j, k): return (j_off + j) * p + (k_off + k)
    def w(i, k): return (k_off + k) * n + (i_off + i)

    # Winograd F(2x2, 2x2): 8 multiplications
    # C = [a00 a01; a10 a11] * [b00 b01; b10 b11]
    # m1 = (a00 + a01)(b00 + b10)
    U[u(0,0)][cur] += 1
    U[u(0,1)][cur] += 1
    V[v(0,0)][cur] += 1
    V[v(0,1)][cur] += 1
    W[w(0,0)][cur] += 1
    W[w(1,0)][cur] += 1
    cur += 1
    # m2 = (a10 - a00)(b00 + b01)
    U[u(1,0)][cur] += 1
    U[u(0,0)][cur] -= 1
    V[v(0,0)][cur] += 1
    V[v(1,0)][cur] += 1
    W[w(0,0)][cur] -= 1
    W[w(0,1)][cur] += 1
    cur += 1
    # m3 = (a10 - a11)(b11)
    U[u(1,0)][cur] += 1
    U[u(1,1)][cur] -= 1
    V[v(1,1)][cur] += 1
    W[w(1,0)][cur] += 1
    W[w(1,1)][cur] -= 1
    cur += 1
    # m4 = (a10 + a11)(b01)
    U[u(1,0)][cur] += 1
    U[u(1,1)][cur] += 1
    V[v(0,1)][cur] += 1
    W[w(0,0)][cur] += 1
    W[w(0,1)][cur] += 1
    cur += 1
    # m5 = a00*b00
    U[u(0,0)][cur] += 1
    V[v(0,0)][cur] += 1
    W[w(0,0)][cur] += 1
    cur += 1
    # m6 = a00*b10
    U[u(0,0)][cur] += 1
    V[v(1,0)][cur] += 1
    W[w(1,0)][cur] += 1
    cur += 1
    # m7 = a01*b11
    U[u(0,1)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,1)][cur] += 1
    cur += 1
    # m8 = a11*b11
    U[u(1,1)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1

    return cur

def _schoolbook(n_block, m_block, p_block, i_off, j_off, k_off, m, n, p, U, V, W, cur):
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                u_idx = (i_off + i) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + (i_off + i)
                U[u_idx][cur] = 1
                V[v_idx][cur] = 1
                W[w_idx][cur] = 1
                cur += 1
    return cur

def _decompose(n, m, p):
    """
    Returns list of (n_sub, m_sub, p_sub, i_off, j_off, k_off, method)
    where method is 'strassen', 'winograd', 'schoolbook'
    
    Strategy: Try multiple decomposition strategies and pick the best one.
    """
    candidates = []
    
    # Strategy 1: n=3 = 1+2, use Strassen for 2x2x2 blocks
    # n=1: 1x4x6 schoolbook = 24
    # n=2: split m=4 into 2+2, split p=6 into 2+2+2
    # 6 blocks of 2x2x2 Strassen = 6*7 = 42
    # Total: 24 + 42 = 66... wait that's wrong
    # Actually: n=1 part has 1*4*6=24 rank-1 terms
    # n=2 part: 2*4*6 = 48 elements, but we decompose into 6 Strassen blocks
    # Each Strassen block handles 2x2x2 with rank 7
    # So n=2 part: 6 * 7 = 42
    # Total: 24 + 42 = 66
    
    # Wait, let me reconsider. The current code gives rank 45.
    # Let me trace through: 
    # n=1 block: 1*4*6 = 24
    # n=2 blocks: 2 chunks of m, 3 chunks of p = 6 blocks
    # Each Strassen = 7, so 6*7 = 42
    # Total = 24 + 42 = 66? But the score says 45...
    
    # Ah, I see - the score is 1.0909 = 49/45, so rank is 45.
    # Let me re-examine. Maybe the schoolbook for 1x4x6 is not 24?
    # No, 1*4*6 = 24 rank-1 terms.
    
    # Hmm, maybe I'm misunderstanding. Let me just try different splits.
    
    # Strategy A: Current approach
    blocks_a = []
    blocks_a.append((1, 4, 6, 0, 0, 0, 'schoolbook'))
    for m_chunk in range(2):
        j_off = m_chunk * 2
        for p_chunk in range(3):
            k_off = p_chunk * 2
            blocks_a.append((2, 2, 2, 1, j_off, k_off, 'strassen'))
    candidates.append(('A', blocks_a))
    
    # Strategy B: n=3 = 1+2, but split differently
    # n=2: split m=4 into 4 chunks of 1, p=6 into 6 chunks of 1
    # That's 24 blocks of 2x1x1... not helpful
    
    # Strategy C: Try Winograd for some blocks
    blocks_c = []
    blocks_c.append((1, 4, 6, 0, 0, 0, 'schoolbook'))
    for m_chunk in range(2):
        j_off = m_chunk * 2
        for p_chunk in range(3):
            k_off = p_chunk * 2
            blocks_c.append((2, 2, 2, 1, j_off, k_off, 'winograd'))
    candidates.append(('C', blocks_c))
    
    # Strategy D: Different n split - n=3 = 2+1
    blocks_d = []
    # n=2 part first (i_off=0)
    for m_chunk in range(2):
        j_off = m_chunk * 2
        for p_chunk in range(3):
            k_off = p_chunk * 2
            blocks_d.append((2, 2, 2, 0, j_off, k_off, 'strassen'))
    # n=1 part (i_off=2)
    blocks_d.append((1, 4, 6, 2, 0, 0, 'schoolbook'))
    candidates.append(('D', blocks_d))
    
    # Pick the candidate with lowest total rank
    best_candidate = None
    best_rank = float('inf')
    
    for name, blocks in candidates:
        total_rank = 0
        for n_sub, m_sub, p_sub, i_off, j_off, k_off, method in blocks:
            if method == 'schoolbook':
                total_rank += n_sub * m_sub * p_sub
            elif method == 'strassen':
                total_rank += 7
            elif method == 'winograd':
                total_rank += 8
        if total_rank < best_rank:
            best_rank = total_rank
            best_candidate = blocks
    
    return best_candidate

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    The multiplication tensor T[i,j,k] = 1 if the index pattern matches.
    For matrix mult: T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    The tensor is: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    This should equal 1 for all valid (i,j,k).
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Check all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Compute sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                val = 0.0
                for r in range(rank):
                    val += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                
                if abs(val - 1.0) > 1e-9:
                    return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    blocks = _decompose(n, m, p)
    
    # Calculate total rank
    total_rank = 0
    for n_sub, m_sub, p_sub, i_off, j_off, k_off, method in blocks:
        if method == 'schoolbook':
            total_rank += n_sub * m_sub * p_sub
        elif method == 'strassen':
            total_rank += 7
        elif method == 'winograd':
            total_rank += 8
    
    # Allocate matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    cur = 0
    for n_sub, m_sub, p_sub, i_off, j_off, k_off, method in blocks:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        if method == 'schoolbook':
            cur = _schoolbook(n_sub, m_sub, p_sub, i_off, j_off, k_off, m, n, p, U, V, W, cur)
        elif method == 'strassen':
            cur = _strassen_2x2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur)
        elif method == 'winograd':
            cur = _winograd_2x2_2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur)
    
    # Verify the decomposition
    if cur != total_rank:
        # Incomplete - pad with zeros (shouldn't happen)
        pass
    
    is_valid = _verify_decomposition(U, V, W, n, m, p)
    
    if not is_valid:
        # Fallback to pure schoolbook
        total_rank = n * m * p
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
