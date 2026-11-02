"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # We aim for rank 18 based on block decomposition analysis.
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Helper to add multiple entries to a rank-1 term
    # u_list: list of (index, value)
    def add_rank1(u_list, v_list, w_list):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_list:
            U[idx][col] = val
        for idx, val in v_list:
            V[idx][col] = val
        for idx, val in w_list:
            W[idx][col] = val
        col += 1

    # Indices for A (3x3): i*m + j
    # Indices for B (3x4): j*p + k
    # Indices for W (4x3): k*n + i

    # --- Block A11 (2x2) x B11 (2x2) -> C11 (2x2) ---
    # A11: i in {0,1}, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    # C11: i in {0,1}, k in {0,1}
    
    # Strassen's 7 terms for 2x2 multiplication
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Local indices in A11: a=0,0; b=0,1; c=1,0; d=1,1
    # Global A indices: 
    # a: 0*3+0=0, b: 0*3+1=1, c: 1*3+0=3, d: 1*3+1=4
    
    # Local indices in B11: e=0,0; f=0,1; g=1,0; h=1,1
    # Global B indices:
    # e: 0*4+0=0, f: 0*4+1=1, g: 1*4+0=4, h: 1*4+1=5
    
    # Local indices in C11: C11=0,0; C12=0,1; C21=1,0; C22=1,1
    # Global W indices (k*n+i):
    # C11 (k=0,i=0): 0*3+0=0
    # C12 (k=1,i=0): 1*3+0=3
    # C21 (k=0,i=1): 0*3+1=1
    # C22 (k=1,i=1): 1*3+1=4
    
    # m1
    add_rank1([(0,1), (4,1)], [(0,1), (5,1)], [(0,1), (4,1)])
    # m2
    add_rank1([(3,1), (4,1)], [(0,1)], [(1,1), (4,-1)])
    # m3
    add_rank1([(0,1)], [(1,1), (5,-1)], [(3,1), (4,1)])
    # m4
    add_rank1([(4,1)], [(4,1), (0,-1)], [(1,1), (4,1)])
    # m5
    add_rank1([(0,1), (1,1)], [(5,1)], [(4,-1), (3,1)])
    # m6
    add_rank1([(3,1), (0,-1)], [(0,1), (1,1)], [(4,1)])
    # m7
    add_rank1([(1,1), (4,-1)], [(4,1), (5,1)], [(0,1)])

    # --- Block A12 (2x1) x B21 (1x2) -> C11 (2x2) ---
    # A12: i in {0,1}, j=2. Global A: 0*3+2=2, 1*3+2=5
    # B21: j=2, k in {0,1}. Global B: 2*4+0=8, 2*4+1=9
    # C11: i in {0,1}, k in {0,1}. Global W: 0, 3, 1, 4
    # Rank 1: Sum_{i,k} A_{i,2} B_{2,k}
    # T_{i,2,k} = 1
    # u_i v_2 w_k
    # u: (2,1), (5,1)
    # v: (8,1), (9,1) -- Wait, v is single index in rank-1?
    # No, rank-1 tensor is u_i v_j w_k.
    # Here j is fixed at 2.
    # So v_2 = 1.
    # u_i = 1 for i in {0,1}.
    # w_k = 1 for k in {0,1}.
    # This is one rank-1 term.
    add_rank1([(2,1), (5,1)], [(8,1), (9,1)], [(0,1), (1,1), (3,1), (4,1)])
    # Note: v indices 8 and 9 correspond to j=2, k=0 and j=2, k=1.
    # In the tensor product, V has indices j*p+k.
    # The term is sum_j V_j. Here only j=2 is non-zero.
    # So V[8]=1, V[9]=1 is correct for the rank-1 decomposition of the block?
    # Wait. The rank-1 tensor for M_{2,1,2} is u_i * v_j * w_k.
    # Since j is fixed to 2, v_2=1, v_other=0.
    # So V[8]=1, V[9]=1 is correct?
    # The index in V is j*p+k.
    # For j=2, k=0: idx 8.
    # For j=2, k=1: idx 9.
    # Yes.
    
    # --- Block A11 (2x2) x B12 (2x1) -> C12 (2x1) ---
    # A11: i in {0,1}, j in {0,1}.
    # B12: j in {0,1}, k=2. Global B: 0*4+2=2, 1*4+2=6
    # C12: i in {0,1}, k=2. Global W: 2*3+0=6, 2*3+1=7
    # Rank 1.
    add_rank1([(0,1), (1,1), (3,1), (4,1)], [(2,1), (6,1)], [(6,1), (7,1)])

    # --- Block A12 (2x1) x B22 (1x1) -> C12 (2x1) ---
    # A12: i in {0,1}, j=2.
    # B22: j=2, k=2. Global B: 2*4+2=10
    # C12: i in {0,1}, k=2.
    # Rank 1.
    add_rank1([(2,1), (5,1)], [(10,1)], [(6,1), (7,1)])

    # --- Block A11 (2x2) x B13 (2x1) -> C13 (2x1) ---
    # B13: j in {0,1}, k=3. Global B: 0*4+3=3, 1*4+3=7
    # C13: i in {0,1}, k=3. Global W: 3*3+0=9, 3*3+1=10
    # Rank 1.
    add_rank1([(0,1), (1,1), (3,1), (4,1)], [(3,1), (7,1)], [(9,1), (10,1)])

    # --- Block A12 (2x1) x B23 (1x1) -> C13 (2x1) ---
    # B23: j=2, k=3. Global B: 2*4+3=11
    # C13: i in {0,1}, k=3.
    # Rank 1.
    add_rank1([(2,1), (5,1)], [(11,1)], [(9,1), (10,1)])

    # --- Block A21 (1x2) x B11 (2x2) -> C21 (1x2) ---
    # A21: i=2, j in {0,1}. Global A: 2*3+0=6, 2*3+1=7
    # B11: j in {0,1}, k in {0,1}.
    # C21: i=2, k in {0,1}. Global W: 0*3+2=2, 1*3+2=5
    # Rank 1.
    add_rank1([(6,1), (7,1)], [(0,1), (1,1), (4,1), (5,1)], [(2,1), (5,1)])

    # --- Block A22 (1x1) x B21 (1x2) -> C21 (1x2) ---
    # A22: i=2, j=2. Global A: 2*3+2=8
    # B21: j=2, k in {0,1}.
    # C21: i=2, k in {0,1}.
    # Rank 1.
    add_rank1([(8,1)], [(8,1), (9,1)], [(2,1), (5,1)])

    # --- Block A21 (1x2) x B12 (2x1) -> C22 (1x1) ---
    # A21: i=2, j in {0,1}.
    # B12: j in {0,1}, k=2.
    # C22: i=2, k=2. Global W: 2*3+2=8
    # Rank 1.
    add_rank1([(6,1), (7,1)], [(2,1), (6,1)], [(8,1)])

    # --- Block A22 (1x1) x B22 (1x1) -> C22 (1x1) ---
    # A22: i=2, j=2.
    # B22: j=2, k=2.
    # C22: i=2, k=2.
    # Rank 1.
    add_rank1([(8,1)], [(10,1)], [(8,1)])

    # --- Block A21 (1x2) x B13 (2x1) -> C23 (1x1) ---
    # A21: i=2, j in {0,1}.
    # B13: j in {0,1}, k=3.
    # C23: i=2, k=3. Global W: 3*3+2=11
    # Rank 1.
    add_rank1([(6,1), (7,1)], [(3,1), (7,1)], [(11,1)])

    # --- Block A22 (1x1) x B23 (1x1) -> C23 (1x1) ---
    # A22: i=2, j=2.
    # B23: j=2, k=3.
    # C23: i=2, k=3.
    # Rank 1.
    add_rank1([(8,1)], [(11,1)], [(11,1)])

    # Check if we used exactly 18 columns
    # If col < 18, we have unused columns (zeros).
    # If col > 18, we overflowed.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
