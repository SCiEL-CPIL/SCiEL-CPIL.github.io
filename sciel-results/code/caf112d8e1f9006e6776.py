"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        
        for (i, j), val in u_vals.items():
            U_row[i * m + j] = val
        for (j, k), val in v_vals.items():
            V_row[j * p + k] = val
        for (k, i), val in w_vals.items():
            W_row[k * n + i] = val
            
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)

    # Block 1: j in {0, 1}
    # Sub-block 1.1: k in {0, 1} (2x2x2 Strassen)
    
    # M1
    add_term(
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    )
    # M2
    add_term(
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1}
    )
    # M3
    add_term(
        {(0,0): 1},
        {(1,0): 1, (0,0): -1},
        {(1,0): 1}
    )
    # M4
    add_term(
        {(1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,1): 1}
    )
    # M5
    add_term(
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(1,0): 1}
    )
    # M6
    add_term(
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(0,1): 1}
    )
    # M7
    add_term(
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(1,0): 1}
    )
    
    # Sub-block 1.2: k in {2, 3, 4} (2x2x3 Schoolbook)
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_term(
                    {(i, j): 1},
                    {(j, k): 1},
                    {(k, i): 1}
                )
                
    # Block 2: j in {2, 3}
    # Sub-block 2.1: k in {0, 1} (2x2x2 Strassen)
    
    # M1
    add_term(
        {(0,2): 1, (1,3): 1},
        {(2,0): 1, (3,1): 1},
        {(0,0): 1, (1,1): 1}
    )
    # M2
    add_term(
        {(1,2): 1, (1,3): 1},
        {(2,0): 1},
        {(0,1): 1}
    )
    # M3
    add_term(
        {(0,2): 1},
        {(3,0): 1, (2,0): -1},
        {(1,0): 1}
    )
    # M4
    add_term(
        {(1,3): 1},
        {(2,1): 1, (3,1): -1},
        {(1,1): 1}
    )
    # M5
    add_term(
        {(0,2): 1, (0,3): 1},
        {(3,1): 1},
        {(1,0): 1}
    )
    # M6
    add_term(
        {(1,2): 1, (0,2): -1},
        {(2,0): 1, (2,1): 1},
        {(0,1): 1}
    )
    # M7
    add_term(
        {(0,3): 1, (1,3): -1},
        {(3,0): 1, (3,1): 1},
        {(1,0): 1}
    )
    
    # Sub-block 2.2: k in {2, 3, 4} (2x2x3 Schoolbook)
    for i in range(2):
        for j in range(2, 4):
            for k in range(2, 5):
                add_term(
                    {(i, j): 1},
                    {(j, k): 1},
                    {(k, i): 1}
                )
                
    # Block 3: j in {4}
    # 2x1x5 Schoolbook
    for i in range(2):
        for k in range(5):
            add_term(
                {(i, 4): 1},
                {(4, k): 1},
                {(k, i): 1}
            )

    rank = len(U_cols)
    
    U_arr = np.array(U_cols).T
    V_arr = np.array(V_cols).T
    W_arr = np.array(W_cols).T
    
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
