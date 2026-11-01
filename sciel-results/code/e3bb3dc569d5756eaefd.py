"""Matrix multiplication 2,2,3 — rank-minimal exact tensor decomposition.

We search for the minimal rank r such that there exist factor matrices
U (n*m, r), V (m*p, r), W (p*n, r) with integer or half-integer entries
satisfying the tensor decomposition identity.

Strategy:
1. Try rank 10: randomized gradient descent to find a solution
2. Fall back to rank 11 (Strassen 2x2x2 + schoolbook 2x2x1)
3. Fall back to rank 12 (pure schoolbook)
"""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 2, 3]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 15.0

def check_time():
    """Return True if we should stop searching."""
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def verify_decomposition(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def get_rank_11_solution():
    """
    Rank 11 solution: Strassen for 2x2x2 (rank 7) + schoolbook for 2x2x1 (rank 4).
    
    Strassen 2x2x2 decomposition (standard):
    For multiplying A (2x2) by B (2x2):
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A10)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    In tensor form, for the 2x2x2 part (k=0,1):
    We need U[i*2+j, c], V[j*2+k, c], W[k*2+i, c] for c=0..6
    
    Let me set up the Strassen factors explicitly.
    A has entries A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3
    B has entries B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,2,3
    C has entries C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0,1,2,3
    
    Strassen factors (U for A, V for B, W for C):
    c=0: M1: U=[1,0,0,1], V=[1,1,1,1], W=[1,0,0,1]  -> (A00+A11)(B00+B11)
    c=1: M2: U=[0,0,1,1], V=[1,0,1,0], W=[0,0,1,0]  -> (A10+A11)B00
    c=2: M3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0] -> A00(B01-B11)
    c=3: M4: U=[0,0,0,1], V=[0,0,1,-1], W=[0,0,1,0] -> A11(B10-B00)
    c=4: M5: U=[1,0,1,0], V=[0,0,0,1], W=[0,0,-1,0] -> (A00+A10)B11
    c=5: M6: U=[0,-1,1,0], V=[1,0,1,0], W=[0,0,0,1] -> (A10-A00)(B00+B10)
    c=6: M7: U=[0,1,0,-1], V=[0,1,0,1], W=[0,1,0,0] -> (A01-A11)(B01+B11)
    
    Wait, let me re-derive. The tensor T[i,j,k] for 2x2x2 is 1 for all i,j,k.
    We need sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all i,j,k.
    
    Let me use the standard Strassen decomposition. The factors are:
    
    For M1 = (A00+A11)(B00+B11):
      U: [1,0,0,1] (A00 + A11)
      V: [1,1,1,1] (B00 + B01 + B10 + B11)... no that's wrong.
      
    Let me be more careful. V[j*2+k, c] where j is row of B, k is col of B.
    V[0,c]=B00 coefficient, V[1,c]=B01 coefficient, V[2,c]=B10 coefficient, V[3,c]=B11 coefficient.
    
    M1 = (A00+A11)(B00+B11):
      U: [1, 0, 0, 1]  (A00 + A11)
      V: [1, 0, 0, 1]  (B00 + B11)
      W: [1, 0, 0, 1]  (C00 + C11)
    
    M2 = (A10+A11)B00:
      U: [0, 0, 1, 1]
      V: [1, 0, 0, 0]
      W: [0, 0, 1, 0]  (C10)
    
    M3 = A00(B01-B11):
      U: [1, 0, 0, 0]
      V: [0, 1, 0, -1]
      W: [0, 1, 0, 0]  (C01)
    
    M4 = A11(B10-B00):
      U: [0, 0, 0, 1]
      V: [0, 0, 1, -1]
      W: [0, 0, 1, 0]  (C10)
    
    M5 = (A00+A10)B11:
      U: [1, 0, 1, 0]
      V: [0, 0, 0, 1]
      W: [0, 0, -1, 0]  (-C10)
    
    M6 = (A10-A00)(B00+B10):
      U: [0, -1, 1, 0]
      V: [1, 0, 1, 0]
      W: [0, 0, 0, 1]  (C11)
    
    M7 = (A01-A11)(B01+B11):
      U: [0, 1, 0, -1]
      V: [0, 1, 0, 1]
      W: [0, 1, 0, 0]  (C01)
    
    Now for the 2x2x1 part (k=2), we need:
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    Schoolbook for this:
    c=7: A00*B02: U=[1,0,0,0], V=[1,0,0,0,0,1] (B02 is index 2 in 2x3, so V[0*3+2]=V[2]), W=[0,0,0,0,0,1] (C02 is index 0*2+0=0... wait)
    
    Let me re-index. For 2x2x3:
    U: 4 x r, indices i*2+j for i=0,1 j=0,1
    V: 6 x r, indices j*3+k for j=0,1 k=0,1,2
    W: 6 x r, indices k*2+i for k=0,1,2 i=0,1
    
    W[k*2+i]: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3; k=2,i=0 -> 4; k=2,i=1 -> 5
    
    For the Strassen part (k=0,1):
    W indices used: k*2+i for k in {0,1}, i in {0,1} -> indices 0,1,2,3
    
    For the 2x2x1 part (k=2):
    W indices: k=2, i=0 -> 4; k=2, i=1 -> 5
    
    c=7: A00*B02 -> C02
      U[0,7]=1 (A00)
      V[2,7]=1 (B02, j=0,k=2 -> 0*3+2=2)
      W[4,7]=1 (C02, k=2,i=0 -> 2*2+0=4)
    
    c=8: A01*B12 -> C02
      U[1,8]=1 (A01)
      V[5,8]=1 (B12, j=1,k=2 -> 1*3+2=5)
      W[4,8]=1 (C02)
    
    c=9: A10*B02 -> C12
      U[2,9]=1 (A10)
      V[2,9]=1 (B02)
      W[5,9]=1 (C12, k=2,i=1 -> 2*2+1=5)
    
    c=10: A11*B12 -> C12
      U[3,10]=1 (A11)
      V[5,10]=1 (B12)
      W[5,10]=1 (C12)
    
    Now let me build the full rank-11 solution.
    """
    rank = 11
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(6)]
    W = [[0.0] * rank for _ in range(6)]
    
    # Strassen factors for 2x2x2 part (k=0,1)
    # c=0: M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # c=1: M2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    
    # c=2: M3
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[1][2] += 1
    
    # c=3: M4
    U[3][3] += 1
    V[2][3] += 1; V[0][3] += -1
    W[2][3] += 1
    
    # c=4: M5
    U[0][4] += 1; U[2][4] += 1
    V[3][4] += 1
    W[2][4] += -1
    
    # c=5: M6
    U[1][5] += -1; U[2][5] += 1
    V[0][5] += 1; V[2][5] += 1
    W[3][5] += 1
    
    # c=6: M7
    U[1][6] += 1; U[3][6] += -1
    V[1][6] += 1; V[3][6] += 1
    W[1][6] += 1
    
    # Schoolbook for 2x2x1 part (k=2)
    # c=7: A00*B02 -> C02
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    
    # c=8: A01*B12 -> C02
    U[1][8] += 1
    V[5][8] += 1
    W[4][8] += 1
    
    # c=9: A10*B02 -> C12
    U[2][9] += 1
    V[2][9] += 1
    W[5][9] += 1
    
    # c=10: A11*B12 -> C12
    U[3][10] += 1
    V[5][10] += 1
    W[5][10] += 1
    
    return {"U": U, "V": V, "W": W}

def get_rank_12_solution():
    """Pure schoolbook rank 12."""
    rank = 12
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j][col] = 1
                V[j*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def try_rank_10_gradient_descent(max_iters=50000, lr=0.01, n_restarts=50):
    """
    Try to find a rank-10 decomposition using randomized gradient descent.
    The tensor T[i,j,k] = 1 for all i in [0,2), j in [0,2), k in [0,3).
    We minimize sum_{i,j,k} (sum_c U[i*2+j,c] * V[j*3+k,c] * W[k*2+i,c] - 1)^2
    """
    n, m, p = 2, 2, 3
    rank = 10
    U_size = n * m  # 4
    V_size = m * p  # 6
    W_size = p * n  # 6
    
    # Precompute index mappings
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    best_loss = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(n_restarts):
        if check_time():
            break
        
        # Random initialization
        U = np.random.randn(U_size, rank) * 0.1
        V = np.random.randn(V_size, rank) * 0.1
        W = np.random.randn(W_size, rank) * 0.1
        
        # Add some structure: make some entries nonzero
        for c in range(rank):
            if random.random() < 0.5:
                U[random.randint(0, U_size-1), c] += 1.0
            if random.random() < 0.5:
                V[random.randint(0, V_size-1), c] += 1.0
            if random.random() < 0.5:
                W[random.randint(0, W_size-1), c] += 1.0
        
        current_loss = float('inf')
        for it in range(max_iters):
            if check_time():
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            loss = 0.0
            
            for (i, j, k) in indices:
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                
                # Compute the dot product
                val = 0.0
                for c in range(rank):
                    val += U[ui, c] * V[vj, c] * W[wk, c]
                
                err = val - 1.0
                loss += err * err
                
                # Gradient with respect to U[ui, c]
                for c in range(rank):
                    dU[ui, c] += 2.0 * err * V[vj, c] * W[wk, c]
                    dV[vj, c] += 2.0 * err * U[ui, c] * W[wk, c]
                    dW[wk, c] += 2.0 * err * U[ui, c] * V[vj, c]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check if we're close enough
            if loss < current_loss:
                current_loss = loss
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
            
            if loss < 1e-10:
                break
            
            # Adaptive learning rate
            if it % 1000 == 0 and it > 0:
                lr *= 0.99
        
        if best_loss < 1e-10:
            break
    
    if best_U is not None and best_loss < 1e-10:
        # Round to nearest half-integer
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify the rounded solution
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        
        if verify_decomposition(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Try rank 10 first (randomized gradient descent)
    sol = try_rank_10_gradient_descent()
    if sol is not None:
        return sol
    
    # Try rank 11 (Strassen + schoolbook)
    sol = get_rank_11_solution()
    if verify_decomposition(sol["U"], sol["V"], sol["W"]):
        return sol
    
    # Fallback to rank 12
    sol = get_rank_12_solution()
    if verify_decomposition(sol["U"], sol["V"], sol["W"]):
        return sol
    
    # Should never get here
    return get_rank_12_solution()
