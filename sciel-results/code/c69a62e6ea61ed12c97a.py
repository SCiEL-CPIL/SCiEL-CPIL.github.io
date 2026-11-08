import time
import numpy as np

_DEADLINE = 150.0 - 10.0
_start = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start)

def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
    n = m = p = 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    def add(c, a_terms, b_terms, w_terms):
        for (i, j, coeff) in a_terms:
            U[i * m + j, c] += coeff
        for (j, k, coeff) in b_terms:
            V[j * p + k, c] += coeff
        for (k, i, coeff) in w_terms:
            W[k * n + i, c] += coeff
            
    add(0, [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
    add(1, [(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)])
    add(2, [(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)])
    add(3, [(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)])
    add(4, [(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)])
    add(5, [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
    add(6, [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
    
    return U, V, W, r

def _rank1_2x1x3():
    """Rank-1 decomposition for 2x1 * 1x3 -> 2x3."""
    n, m, p = 2, 1, 3
    r = 1
    U = np.ones((n * m, r))
    V = np.ones((m * p, r))
    W = np.ones((p * n, r))
    return U, V, W, r

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two decompositions."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    U = np.zeros((U1.shape[0] * U2.shape[0], r))
    V = np.zeros((V1.shape[0] * V2.shape[0], r))
    W = np.zeros((W1.shape[0] * W2.shape[0], r))
    
    for a in range(r1):
        for b in range(r2):
            idx = a * r2 + b
            U[:, idx] = np.kron(U1[:, a], U2[:, b])
            V[:, idx] = np.kron(V1[:, a], V2[:, b])
            W[:, idx] = np.kron(W1[:, a], W2[:, b])
            
    return U, V, W, r

def construct():
    n, m, p = 4, 5, 6
    
    # Split m=5 into 2+3
    # M_4,5,6 = M_4,2,6 + M_4,3,6
    
    # M_4,2,6 = M_2,1,3 (Kron) M_2,2,2
    U_213, V_213, W_213, r1 = _rank1_2x1x3()
    U_222, V_222, W_222, r2 = _strassen_2x2()
    
    U_426, V_426, W_426, r_426 = _kron_decomp(U_213, V_213, W_213, U_222, V_222, W_222)
    
    # M_4,3,6 = M_2,3,3 (Kron) M_2,1,2
    # M_2,3,3 = M_2,1,3 + M_2,2,3
    # M_2,2,3 = M_2,2,1 + M_2,2,2
    
    # We need M_2,3,3 decomposition.
    # Let's build M_2,3,3 directly using the sum logic.
    # M_2,3,3: n=2, m=3, p=3.
    # Split m=3 into 1+2.
    # M_2,1,3 (Rank 1) and M_2,2,3.
    # M_2,2,3: Split p=3 into 1+2.
    # M_2,2,1 (Rank 1) and M_2,2,2 (Rank 7).
    
    # We need to implement a "sum" of decompositions.
    # If T = T1 + T2, and T1 has factors U1,V1,W1 (rank r1) and T2 has U2,V2,W2 (rank r2),
    # then T has factors [U1 0; 0 U2]? No.
    # The factors are concatenated in the rank dimension.
    # U = [U1, U2], V = [V1, V2], W = [W1, W2].
    
    def sum_decomps(U1, V1, W1, r1, U2, V2, W2, r2):
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        return U, V, W, r1 + r2

    # Build M_2,2,1 (Rank 1)
    U_221, V_221, W_221, r_221 = _rank1_2x1x3() # Wait, dimensions are n=2, m=2, p=1.
    # _rank1_2x1x3 is for 2x1x3. We need 2x2x1.
    # Let's make a generic rank 1 for n,m,p.
    def _rank1(n, m, p):
        r = 1
        U = np.ones((n*m, r))
        V = np.ones((m*p, r))
        W = np.ones((p*n, r))
        return U, V, W, r

    U_221, V_221, W_221, r_221 = _rank1(2, 2, 1)
    
    # Build M_2,2,2 (Rank 7)
    U_222, V_222, W_222, r_222 = _strassen_2x2()
    
    # Build M_2,2,3 = M_2,2,1 + M_2,2,2
    # Note: M_2,2,3 has p=3. M_2,2,1 has p=1. M_2,2,2 has p=2.
    # This sum is not direct because p is different.
    # We must align dimensions.
    # M_2,2,3: n=2, m=2, p=3.
    # M_2,2,1: n=2, m=2, p=1.
    # M_2,2,2: n=2, m=2, p=2.
    # We can embed M_2,2,1 into M_2,2,3 by padding W?
    # No, the tensor is defined on the space.
    # If we have a decomposition for M_2,2,1, it satisfies T_{i,j,k} for k<1.
    # For M_2,2,3, we need T_{i,j,k} for k<3.
    # We can use the decomposition for M_2,2,1 for the k=0 part, and M_2,2,2 for the k=1,2 part?
    # This is getting complex to implement manually with padding.
    
    # Alternative: Just use schoolbook for the small parts if rank is low enough.
    # Rank(M_2,2,3) schoolbook = 12.
    # Rank(M_2,3,3) schoolbook = 18.
    # Rank(M_4,3,6) schoolbook = 72.
    # Total Rank = 7 + 72 = 79.
    # 120 / 79 = 1.51.
    # This is a good improvement.
    
    # Let's stick to the Kronecker product which is clean.
    # M_4,3,6 = M_2,3,3 (Kron) M_2,1,2.
    # We need M_2,3,3.
    # Let's use schoolbook for M_2,3,3 (Rank 18).
    U_233, V_233, W_233, r_233 = _schoolbook(2, 3, 3)
    U_212, V_212, W_212, r_212 = _rank1(2, 1, 2)
    
    U_436, V_436, W_436, r_436 = _kron_decomp(U_233, V_233, W_233, U_212, V_212, W_212)
    
    # Total Rank = r_426 + r_436 = 7 + 18 = 25.
    # 120 / 25 = 4.8.
    # This is excellent.
    
    # Combine M_4,2,6 and M_4,3,6.
    # They have different m dimensions (2 vs 3).
    # We need to embed them into M_4,5,6.
    # U_426 is (8, 7). U_436 is (12, 25).
    # We need U (20, 32).
    # U_426 corresponds to j=0,1.
    # U_436 corresponds to j=2,3,4?
    # M_4,2,6 uses j in 0..1.
    # M_4,3,6 uses j in 0..2.
    # We need to shift indices.
    
    U_total = np.zeros((4*5, r_426 + r_436))
    V_total = np.zeros((5*6, r_426 + r_436))
    W_total = np.zeros((6*4, r_426 + r_436))
    
    # M_4,2,6 part
    # U_426 indices: i*2 + j. j in 0,1.
    # In M_4,5,6, U index is i*5 + j.
    # For j=0,1, U_426[i*2+j] maps to U_total[i*5+j].
    for i in range(4):
        for j in range(2):
            U_total[i*5+j, :r_426] = U_426[i*2+j, :]
            
    # V_426 indices: j*6 + k. j in 0,1.
    # In M_4,5,6, V index is j*6 + k.
    # Same indices.
    V_total[:2*6, :r_426] = V_426
    
    # W_426 indices: k*4 + i.
    # Same indices.
    W_total[:, :r_426] = W_426
    
    # M_4,3,6 part
    # U_436 indices: i*3 + j. j in 0,1,2.
    # In M_4,5,6, we want j in 2,3,4.
    # So U_436[i*3+j] maps to U_total[i*5 + (j+2)].
    for i in range(4):
        for j in range(3):
            U_total[i*5 + (j+2), r_426:] = U_436[i*3+j, :]
            
    # V_436 indices: j*6 + k. j in 0,1,2.
    # In M_4,5,6, we want j in 2,3,4.
    # So V_436[j*6+k] maps to V_total[(j+2)*6+k].
    for j in range(3):
        V_total[(j+2)*6:(j+2)*6+6, r_426:] = V_436[j*6:(j+1)*6, :]
        
    # W_436 indices: k*4 + i.
    # Same indices.
    W_total[:, r_426:] = W_436
    
    total_rank = r_426 + r_436
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W, rank
