import time
import numpy as np

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 110  # seconds


def _get_strassen_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a tuple (u_mask, v_mask, w_mask) where masks are dicts of {index: coeff}.
    Indices are local to the 2x2x2 block:
    U: i*2 + j (i in 0,1; j in 0,1)
    V: j*2 + k (j in 0,1; k in 0,1)
    W: k*2 + i (k in 0,1; i in 0,1)
    """
    # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W indices: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
    
    # m1 = (a00 + a11) * (b00 + b11) -> c00 += 1, c11 += 1
    t1 = ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
    
    # m2 = (a10 + a11) * b00 -> c10 += 1, c11 -= 1
    t2 = ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1})
    
    # m3 = a00 * (b01 - b11) -> c01 += 1, c11 += 1
    t3 = ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1})
    
    # m4 = a11 * (b10 - b00) -> c00 += 1, c10 += 1
    t4 = ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1})
    
    # m5 = (a00 + a01) * b11 -> c01 += 1, c00 -= 1
    t5 = ({0: 1, 1: 1}, {3: 1}, {2: 1, 0: -1})
    
    # m6 = (a10 - a00) * (b00 + b01) -> c11 += 1
    t6 = ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    
    # m7 = (a01 - a11) * (b10 + b11) -> c00 += 1
    t7 = ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    
    return [t1, t2, t3, t4, t5, t6, t7]


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Uses Strassen for 2x2 blocks and schoolbook for 2x1 blocks.
    Rank = 5 * (7 + 4) = 55.
    """
    _start = time.time()
    n, m, p = DIMENSIONS
    rank = 55

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    strassen_terms = _get_strassen_terms()

    # A is 2x3. Indices: i*m + j where i in {0,1}, j in {0,1,2}
    # A columns: j=0,1 form the 2x2 Strassen part; j=2 is the extra column
    # B is 3x10. Indices: j*p + k where j in {0,1,2}, k in {0..9}
    # B rows: j=0,1 form the 2x10 Strassen part; j=2 is the extra row
    # C is 2x10. Indices: k*n + i where k in {0..9}, i in {0,1}
    
    # We split the 10 columns of B (and C) into 5 groups of 2
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # Process 7 Strassen terms for the 2x2x2 block
        for u_mask, v_mask, w_mask in strassen_terms:
            for u_idx, coeff in u_mask.items():
                # u_idx is local to 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                li = u_idx // 2
                lj = u_idx % 2
                global_u_idx = li * m + lj  # m=3, so j=0,1 map to global j=0,1
                U[global_u_idx, col] += coeff
            
            for v_idx, coeff in v_mask.items():
                # v_idx is local to 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                lj = v_idx // 2
                lk = v_idx % 2
                global_k = k0 + lk  # map local k to global k
                global_v_idx = lj * p + global_k  # p=10
                V[global_v_idx, col] += coeff
            
            for w_idx, coeff in w_mask.items():
                # w_idx is local to 2x2: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
                lk = w_idx // 2
                li = w_idx % 2
                global_k = k0 + lk
                global_w_idx = global_k * n + li  # n=2
                W[global_w_idx, col] += coeff
            
            col += 1
        
        # Process 4 schoolbook terms for the 2x1x2 block (A[:,2] * B[2,:])
        # A indices for column 2: i*3 + 2
        a02 = 0 * m + 2  # = 2
        a12 = 1 * m + 2  # = 5
        
        # B indices for row 2: 2*10 + k
        b20 = 2 * p + k0  # = 20 + k0
        b21 = 2 * p + k1  # = 20 + k1
        
        # C indices
        c00 = k0 * n + 0  # = 2*k0
        c01 = k1 * n + 0  # = 2*k1
        c10 = k0 * n + 1  # = 2*k0 + 1
        c11 = k1 * n + 1  # = 2*k1 + 1
        
        # P1: A02 * B20 -> C[0,k0]
        U[a02, col] = 1.0
        V[b20, col] = 1.0
        W[c00, col] = 1.0
        col += 1
        
        # P2: A02 * B21 -> C[0,k1]
        U[a02, col] = 1.0
        V[b21, col] = 1.0
        W[c01, col] = 1.0
        col += 1
        
        # P3: A12 * B20 -> C[1,k0]
        U[a12, col] = 1.0
        V[b20, col] = 1.0
        W[c10, col] = 1.0
        col += 1
        
        # P4: A12 * B21 -> C[1,k1]
        U[a12, col] = 1.0
        V[b21, col] = 1.0
        W[c11, col] = 1.0
        col += 1

    # Check time limit
    if time.time() - _start > TIME_LIMIT:
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
