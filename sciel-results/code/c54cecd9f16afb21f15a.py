import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s
    
    n, m, p = 2, 2, 13
    rank = 46
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(26)]
    W = [[0] * rank for _ in range(26)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
        
    # Check deadline
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        add_term({0: 1, 3: 1}, {k0: 1, 13+k1: 1}, {2*k0: 1, 2*k1+1: 1})
        add_term({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1})
        add_term({0: 1}, {k1: 1, 13+k1: -1}, {2*k1: 1, 2*k1+1: 1})
        add_term({3: 1}, {13+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1})
        add_term({0: 1, 1: 1}, {13+k1: 1}, {2*k0: -1, 2*k1: 1})
        add_term({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1})
        add_term({1: 1, 3: -1}, {13+k0: 1, 13+k1: 1}, {2*k0: 1})
        
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}

    k = 12
    add_term({0: 1}, {k: 1}, {2*k: 1})
    add_term({1: 1}, {13+k: 1}, {2*k: 1})
    add_term({2: 1}, {k: 1}, {2*k+1: 1})
    add_term({3: 1}, {13+k: 1}, {2*k+1: 1})
    
    return {"U": U, "V": V, "W": W}
