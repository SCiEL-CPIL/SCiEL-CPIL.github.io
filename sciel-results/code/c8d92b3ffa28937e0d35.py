"""Matrix multiplication (2,3,9) rank search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p, r):
    # U:(n*m,r) V:(m*p,r) W:(p*n,r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _verify_fast(U, V, W, n, m, p, r):
    """Fast verification using matrix multiplication."""
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is equivalent to: for each c, outer product of U[:,c], V[:,c], W[:,c]
    # Let's use a smarter approach:
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    # T = sum_c U[:,:,c] * V[:,:,c] * W[:,:,c] (elementwise in c, broadcast in i,j,k)
    # Actually: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    
    # For each c, compute the contribution to T
    # T[i,j,k] += Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # This is a sum over c of a rank-1 tensor
    
    # We can compute this as:
    # For each j, sum_c Ur[:,j,c] * Vr[j,:,c] * Wr[:,:,j] ... no that's not right
    
    # Let's just use einsum which is optimized
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7 decomposition.
    """
    # Strassen's 7 multiplications for 2x2x2
    # U: (2*2, 7) = (4, 7), V: (2*2, 7) = (4, 7), W: (2*2, 7) = (4, 7)
    n = m = p = 2
    r = 7
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    
    # U indices: (i,j) -> i*2+j, i in {0,1}, j in {0,1}
    # V indices: (j,k) -> j*2+k, j in {0,1}, k in {0,1}
    # W indices: (k,i) -> k*2+i, k in {0,1}, i in {0,1}
    
    # M1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    # W: C00 += M1, C11 += M1
    W[0, 0] = 1  # C00 (k=0,i=0)
    W[3, 0] = 1  # C11 (k=1,i=1)
    
    # M2: (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    # W: C10 += M2, C11 -= M2
    W[2, 1] = 1  # C10 (k=1,i=0)
    W[3, 1] = -1 # C11 (k=1,i=1)
    
    # M3: A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    # W: C01 += M3, C11 += M3
    W[1, 2] = 1  # C01 (k=0,i=1)
    W[3, 2] = 1  # C11 (k=1,i=1)
    
    # M4: A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    # W: C00 += M4, C10 += M4
    W[0, 3] = 1  # C00 (k=0,i=0)
    W[2, 3] = 1  # C10 (k=1,i=0)
    
    # M5: (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    # W: C00 -= M5, C01 += M5
    W[0, 4] = -1 # C00 (k=0,i=0)
    W[1, 4] = 1  # C01 (k=0,i=1)
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    # W: C11 += M6
    W[3, 5] = 1  # C11 (k=1,i=1)
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    # W: C00 += M7
    W[0, 6] = 1  # C00 (k=0,i=0)
    
    return U, V, W

def _construct_2x3x9_from_blocks():
    """
    Construct 2x3x9 decomposition using block structure.
    Split the 3x9 matrix into three 3x3 blocks.
    For each 2x3x3 block, use a decomposition.
    
    For 2x3x3: split into 2x2x3 and 2x1x3
    2x2x3: use Strassen for the 2x2x2 part + schoolbook for 2x2x1
    2x1x3: rank 3 (outer product structure)
    
    Actually, let's think more carefully.
    2x3x3: C[i,k] = sum_j A[i,j] B[j,k]
    Split j into j=0,1 (2x2x3) and j=2 (2x1x3)
    
    2x2x3: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    This is like multiplying a 2x2 by a 2x3.
    We can view this as: for each k, it's a 2x2x1 multiplication.
    Or we can use the fact that it's a "fat" matrix multiplication.
    
    Let's use a different approach:
    2x3x3 = sum of 3 rank contributions, one for each k
    For fixed k: C[:,k] = A * B[:,k] (2x3 times 3x1 = 2x1)
    This is 2x3x1 multiplication, which has rank 3 (schoolbook).
    So 2x3x3 has rank at most 9 (schoolbook).
    
    But we can do better. Let's use the block structure:
    2x3x3: split A into [A_left (2x2), A_right (2x1)]
    C = A_left * B_left + A_right * B_right
    where B_left is 2x3, B_right is 1x3.
    
    A_left * B_left is 2x2x3: rank at most 6 (schoolbook)
    A_right * B_right is 2x1x3: rank at most 3 (schoolbook)
    Total: 9 (no improvement over schoolbook)
    
    Let's try a different split:
    2x3x3: use Strassen-like approach for the 2x2 part
    A = [A11 A12; A21 A22] * [B1; B2] where B1 is 1x3, B2 is 2x3
    C = A11*B1 + A12*B2 + A21*B1 + A22*B2
       = (A11+A21)*B1 + A12*B2 + A21*B2 ... no this doesn't simplify well
    
    Let me just use the schoolbook for 2x3x3 (rank 9) and see if we can find
    a better decomposition through random search.
    
    For 2x3x9: 3 blocks of 2x3x3, each rank 9, total rank 27.
    """
    n, m, p = 2, 3, 9
    R = 27  # 3 * 9
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # For each of the 3 blocks (corresponding to k=0,1,2 in the 3x3 sub-blocks)
    for g in range(3):
        # Block g handles columns 3g to 3g+2 of the 3x9 matrix
        for i in range(n):
            for j in range(m):
                for k_local in range(3):
                    k = 3 * g + k_local
                    col = 9 * g + (i * m + j) * 3 + k_local  # This is wrong
                    
    # Let me redo this more carefully
    # Schoolbook for 2x3x9: rank = 2*3*9 = 54
    # Block decomposition: 3 blocks of 2x3x3, each rank 9, total 27
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    for g in range(3):  # block index
        for i in range(n):  # row of A
            for j in range(m):  # col of A / row of B
                for k_local in range(3):  # local col of B
                    k = 3 * g + k_local  # global col of B
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_233(deadline, max_trials_per_rank=5000):
    """Search for a rank decomposition of 2x3x3."""
    n, m, p = 2, 3, 3
    best_rank = n * m * p
    best_solution = None
    
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
        [0, 1, -1, 2, -2],
        [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5],
    ]
    
    # Lower bound for 2x3x3 is max(6, 9, 6) = 9
    for r in range(9, best_rank):
        if time.time() > deadline:
            break
        trials = max_trials_per_rank
        for _ in range(trials):
            if time.time() > deadline:
                break
            vs = random.choice(val_sets)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            for c in range(r):
                for idx in range(n * m):
                    if random.random() < 0.4:
                        U[idx, c] = random.choice(vs)
                for idx in range(m * p):
                    if random.random() < 0.4:
                        V[idx, c] = random.choice(vs)
                for idx in range(p * n):
                    if random.random() < 0.4:
                        W[idx, c] = random.choice(vs)
                        
            if _verify(U, V, W, n, m, p, r):
                best_rank = r
                best_solution = (U, V, W)
                break
    
    if best_solution:
        return best_solution
    return None

def _construct_from_233(U233, V233, W233, r233):
    """Construct 2x3x9 decomposition from 2x3x3."""
    n, m, p = 2, 3, 9
    R = 3 * r233
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for g in range(3):
        for c in range(r233):
            col = 3 * c + g
            for i in range(2):
                for j in range(3):
                    U[i * 3 + j, col] = U233[i, j, c]
            
            for j in range(3):
                for l in range(3):
                    k = 3 * g + l
                    V[j * 9 + k, col] = V233[j, l, c]
            
            for l in range(3):
                k = 3 * g + l
                for i in range(2):
                    W[k * 2 + i, col] = W233[l, i, c]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _direct_search_239(deadline, best_rank, max_trials_per_rank=2000):
    """Direct random search for 2x3x9 decomposition."""
    n, m, p = 2, 3, 9
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
    ]
    
    for r in range(max(27, best_rank - 5), best_rank):
        if time.time() > deadline:
            break
        trials = max_trials_per_rank
        for _ in range(trials):
            if time.time() > deadline:
                break
            vs = random.choice(val_sets)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                for idx in range(n * m):
                    if random.random() < 0.5:
                        U[idx, c] = random.choice(vs)
                for idx in range(m * p):
                    if random.random() < 0.5:
                        V[idx, c] = random.choice(vs)
                for idx in range(p * n):
                    if random.random() < 0.5:
                        W[idx, c] = random.choice(vs)
            if _verify(U, V, W, n, m, p, r):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, r
    
    return None, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try block decomposition (2x3x9 = 3 * 2x3x3, each rank 9 -> total 27)
    block_sol = _construct_2x3x9_from_blocks()
    block_rank = 27
    if block_rank < best_rank:
        best = block_sol
        best_rank = block_rank

    # Try to find a better solution using the 2x3x3 block structure
    sol_233 = _search_233(deadline)
    if sol_233:
        U233, V233, W233 = sol_233
        r233 = U233.shape[1]
        candidate = _construct_from_233(U233, V233, W233, r233)
        r_candidate = 3 * r233
        if r_candidate < best_rank:
            best = candidate
            best_rank = r_candidate

    # Fallback: direct search for 2x3x9 if time permits
    if time.time() < deadline - 10:
        candidate, r_candidate = _direct_search_239(deadline, best_rank)
        if candidate is not None and r_candidate < best_rank:
            best = candidate
            best_rank = r_candidate

    return best
# EVOLVE-BLOCK-END
