import time
import numpy as np
from scipy.linalg import svd

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0
    
    # Define Strassen pattern for 2x2*2x2
    # Each entry is (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    # Indices are local to the 2x2 block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_pattern = [
        # m1 = (a+d)(e+h)
        {
            'u': [((0,0), 1), ((1,1), 1)],
            'v': [((0,0), 1), ((1,1), 1)],
            'w': [((0,0), 1), ((1,1), 1)]
        },
        # m2 = (c+d)*e
        {
            'u': [((1,0), 1), ((1,1), 1)],
            'v': [((0,0), 1)],
            'w': [((1,0), 1), ((1,1), -1)]
        },
        # m3 = a*(f-h)
        {
            'u': [((0,0), 1)],
            'v': [((0,1), 1), ((1,1), -1)],
            'w': [((0,1), 1), ((1,1), 1)]
        },
        # m4 = d*(g-e)
        {
            'u': [((1,1), 1)],
            'v': [((1,0), 1), ((0,0), -1)],
            'w': [((0,0), 1), ((1,0), 1)]
        },
        # m5 = (a+b)*h
        {
            'u': [((0,0), 1), ((0,1), 1)],
            'v': [((1,1), 1)],
            'w': [((0,0), -1), ((0,1), 1)]
        },
        # m6 = (c-a)*(e+f)
        {
            'u': [((1,0), 1), ((0,0), -1)],
            'v': [((0,0), 1), ((0,1), 1)],
            'w': [((1,1), 1)]
        },
        # m7 = (b-d)*(g+h)
        {
            'u': [((0,1), 1), ((1,1), -1)],
            'v': [((1,0), 1), ((1,1), 1)],
            'w': [((0,0), 1)]
        }
    ]

    terms = [] # List of dicts: {'u': [(global_idx, coeff), ...], ...}
    
    def add_term(u_list, v_list, w_list):
        terms.append({'u': u_list, 'v': v_list, 'w': w_list})

    # 1. Strassen blocks for A[0:2, 0:2] * B[0:2, c:c+2]
    for c in range(0, 8, 2):
        for pat in strassen_pattern:
            u_g = []
            v_g = []
            w_g = []
            
            for (li, lj), coeff in pat['u']:
                global_idx = (li, lj) # 0:0-1, 1:0-1
                u_g.append(((global_idx[0], global_idx[1]), coeff))
                
            for (li, lj), coeff in pat['v']:
                # B local col is c + lj
                global_idx = (li, c + lj)
                v_g.append(((global_idx[0], global_idx[1]), coeff))
                
            for (li, lj), coeff in pat['w']:
                # C local col is c + lj
                global_idx = (li, c + lj)
                w_g.append(((global_idx[0], global_idx[1]), coeff))
                
            add_term(u_g, v_g, w_g)

    # 2. Direct terms
    # A[0:2, 2] * B[2, :]
    for i in range(2):
        for k in range(8):
            add_term(
                [((i, 2), 1)],
                [((2, k), 1)],
                [((i, k), 1)]
            )
            
    # A[2, 0:2] * B[0:2, :]
    for j in range(2):
        for k in range(8):
            add_term(
                [((2, j), 1)],
                [((j, k), 1)],
                [((2, k), 1)]
            )
            
    # A[2, 2] * B[2, :]
    for k in range(8):
        add_term(
            [((2, 2), 1)],
            [((2, k), 1)],
            [((2, k), 1)]
        )
        
    rank = len(terms)
    
    # Build dense matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (i, j), coeff in terms[idx]['u']:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in terms[idx]['v']:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in terms[idx]['w']:
            W[k * n + i][idx] = coeff
            
    # Optimization: Check for linear dependencies in the term set
    # Construct tensor T of shape (n*m, m*p, p*n)
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # This is equivalent to sum_r (U[:,r] \otimes V[:,r] \otimes W[:,r])
    
    # We try to remove columns if they are linearly dependent in the tensor space.
    # This is hard to do directly. 
    # Instead, we rely on the minimal construction of Strassen.
    
    # Convert to nice lists
    def to_nice_list(mat):
        res = []
        for row in mat:
            nrow = []
            for v in row:
                if abs(v - round(v*2)/2) < 1e-9:
                    nv = round(v*2)/2
                    if abs(nv - round(nv)) < 1e-9:
                        nrow.append(int(round(nv)))
                    else:
                        nrow.append(nv)
                else:
                    nrow.append(v)
            res.append(nrow)
        return res

    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
# EVOLVE-BLOCK-END
