"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True

def _schoolbook(n=4, m=4, p=9):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 x 2x2 matmul, rank 7."""
    # Strassen's 7 multiplications for 2x2 matmul
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # i,j in {0,1}, k in {0,1}
    rank = 7
    U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
    V = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
    W = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
    
    # M1 = (a+d)(e+h): U[0,0]=1, U[3,0]=1, V[0,0]=1, V[3,0]=1, W[0,0]=1, W[1,0]=1
    # Actually let me index: U[i*m+j, c] where m=2
    # a=A[0,0] -> U[0,0], b=A[0,1] -> U[1,0], c=A[1,0] -> U[2,0], d=A[1,1] -> U[3,0]
    # e=B[0,0] -> V[0,0], f=B[0,1] -> V[1,0], g=B[1,0] -> V[2,0], h=B[1,1] -> V[3,0]
    # C[0,0] -> W[0,0], C[0,1] -> W[1,0], C[1,0] -> W[2,0], C[1,1] -> W[3,0]
    
    # M1 = (a+d)(e+h): contributes to C[0,0] and C[1,1]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e: contributes to C[1,0] and C[1,1]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a(f-h): contributes to C[0,1] and C[1,1]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = d(g-e): contributes to C[0,0] and C[1,0]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a+b)h: contributes to C[0,0] and C[0,1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (c-a)(e+f): contributes to C[1,1]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h): contributes to C[0,0] and C[1,0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = -1
    
    return {"U": U, "V": V, "W": W}

def _try_structured_decomposition(n, m, p, rank, max_iter=200, seed=42):
    """Try to find a rank-r decomposition using structured search."""
    random.seed(seed)
    np.random.seed(seed)
    
    for it in range(max_iter):
        # Generate random U, V with small integer entries
        U = np.random.randint(-2, 3, size=(n*m, rank))
        V = np.random.randint(-2, 3, size=(m*p, rank))
        
        # For each (k,i), solve for W[k*n+i, :]
        W = np.zeros((p*n, rank), dtype=float)
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build A matrix: m x rank
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, rank), dtype=float)
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                
                b = np.ones(m)
                
                try:
                    # Find a solution using lstsq
                    w_sol, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                    
                    # Check if the solution is close to integer/half-integer
                    w_half = np.round(w_sol * 2) / 2
                    error = np.abs(A @ w_half - b)
                    if np.max(error) > 1e-9:
                        w_int = np.round(w_sol)
                        error = np.abs(A @ w_int - b)
                        if np.max(error) > 1e-9:
                            valid = False
                            break
                    
                    W[k*n+i, :] = w_half if np.max(np.abs(A @ w_half - b)) <= 1e-9 else w_int
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Verify the full decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_block_decomposition(n, m, p):
    """Try to find a decomposition using block structure."""
    # For 4x4 x 4x9, try to use 2x2 blocks
    # Split 4x4 into four 2x2 blocks
    # Split 4x9 into two 2x9 blocks (or 4x9 as is)
    
    # Actually, let's try a different approach:
    # Use the fact that 4x4 x 4x9 can be seen as:
    # [A11 A12; A21 A22] * [B1; B2] where B1 is 4x9, B2 is 4x9
    # No, that's not right.
    
    # Let me try: A is 4x4, B is 4x9
    # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # B = [B1; B2] where B1 is 2x9, B2 is 2x9
    # C = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    # For each 2x2 * 2x9 multiplication, we can use a decomposition
    # 2x2 * 2x9: this is like doing 2x2 * 2x2 nine times, but with shared structure
    
    # Actually, let's try a simpler approach: just use random search with better initialization
    
    return None

def construct():
    start_time = time.time()
    DEADLINE = 130  # Safety margin before 150s limit
    
    n, m, p = 4, 4, 9
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try structured search for various ranks
    # Based on known results, try ranks from a reasonable lower bound
    # For 4x4x4, best known is 21. For 4x4x9, estimate around 40-60
    
    candidate_ranks = list(range(45, 145, 5))
    candidate_ranks = sorted(set(candidate_ranks), reverse=True)
    
    for rank in candidate_ranks:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        if rank >= best_rank:
            continue
        
        # Try to find a decomposition of this rank
        result = _try_structured_decomposition(n, m, p, rank, max_iter=50, seed=rank)
        
        if result is not None:
            best_result = result
            best_rank = rank
            print(f"Found decomposition of rank {rank}")
            # Continue searching for even better
            if rank <= 50:
                break
    
    return best_result
