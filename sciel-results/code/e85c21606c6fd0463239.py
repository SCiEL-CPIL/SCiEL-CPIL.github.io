import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    
    # We use a block-diagonal Strassen decomposition.
    # Partition the 16 columns of B into 8 blocks of 2 columns.
    # For each block, we use the rank-7 Strassen algorithm for 2x2 matrix multiplication.
    # Total rank = 8 * 7 = 56.
    
    total_rank = 56
    
    # Initialize U, V, W as NumPy arrays for efficiency
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Strassen patterns for 2x2 multiplication
    # Indices:
    # A: [[a, b], [c, d]] -> indices 0, 1, 2, 3
    # B: [[e, f], [g, h]] -> indices 0, 1, 2, 3 (local to block)
    # C: [[i, j], [k, l]] -> indices 0, 1, 2, 3 (local to block)
    
    # Strassen formulas:
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * e
    # p3 = a * (f - h)
    # p4 = d * (g - e)
    # p5 = (a + b) * h
    # p6 = (c - a) * (e + f)
    # p7 = (b - d) * (g + h)
    #
    # i = p1 + p4 - p5 + p7
    # j = p3 + p5
    # k = p2 + p4
    # l = p1 - p2 + p3 + p6
    
    # We define the coefficients for each of the 7 products.
    # Each product is a rank-1 tensor (outer product of U_vec, V_vec, W_vec).
    # U_vec is in R^4 (A elements)
    # V_vec is in R^4 (B elements, local)
    # W_vec is in R^4 (C elements, local)
    
    # Pattern 1: p1 = (a + d) * (e + h)
    # U: a=1, d=1 -> [1, 0, 0, 1]
    # V: e=1, h=1 -> [1, 0, 0, 1]
    # W: i=1, l=1 -> [1, 0, 0, 1]
    
    # Pattern 2: p2 = (c + d) * e
    # U: c=1, d=1 -> [0, 0, 1, 1]
    # V: e=1 -> [1, 0, 0, 0]
    # W: k=1, l=-1 -> [0, 0, 1, -1]
    
    # Pattern 3: p3 = a * (f - h)
    # U: a=1 -> [1, 0, 0, 0]
    # V: f=1, h=-1 -> [0, 1, 0, -1]
    # W: j=1, l=1 -> [0, 1, 0, 1]
    
    # Pattern 4: p4 = d * (g - e)
    # U: d=1 -> [0, 0, 0, 1]
    # V: g=1, e=-1 -> [-1, 0, 1, 0]
    # W: i=1, k=1 -> [1, 0, 1, 0]
    
    # Pattern 5: p5 = (a + b) * h
    # U: a=1, b=1 -> [1, 1, 0, 0]
    # V: h=1 -> [0, 0, 0, 1]
    # W: i=-1, j=1 -> [-1, 1, 0, 0]
    
    # Pattern 6: p6 = (c - a) * (e + f)
    # U: c=1, a=-1 -> [-1, 0, 1, 0]
    # V: e=1, f=1 -> [1, 1, 0, 0]
    # W: l=1 -> [0, 0, 0, 1]
    
    # Pattern 7: p7 = (b - d) * (g + h)
    # U: b=1, d=-1 -> [0, 1, 0, -1]
    # V: g=1, h=1 -> [0, 0, 1, 1]
    # W: i=1 -> [1, 0, 0, 0]
    
    patterns = [
        {
            'U': [1, 0, 0, 1],
            'V': [1, 0, 0, 1],
            'W': [1, 0, 0, 1]
        },
        {
            'U': [0, 0, 1, 1],
            'V': [1, 0, 0, 0],
            'W': [0, 0, 1, -1]
        },
        {
            'U': [1, 0, 0, 0],
            'V': [0, 1, 0, -1],
            'W': [0, 1, 0, 1]
        },
        {
            'U': [0, 0, 0, 1],
            'V': [-1, 0, 1, 0],
            'W': [1, 0, 1, 0]
        },
        {
            'U': [1, 1, 0, 0],
            'V': [0, 0, 0, 1],
            'W': [-1, 1, 0, 0]
        },
        {
            'U': [-1, 0, 1, 0],
            'V': [1, 1, 0, 0],
            'W': [0, 0, 0, 1]
        },
        {
            'U': [0, 1, 0, -1],
            'V': [0, 0, 1, 1],
            'W': [1, 0, 0, 0]
        }
    ]
    
    num_blocks = p // 2 # 8 blocks
    
    for q in range(num_blocks):
        for s in range(7):
            col = q * 7 + s
            pat = patterns[s]
            
            # U part: global index i*m + j
            # A is 2x2. i in 0..1, j in 0..1.
            # pat['U'] is [a, b, c, d] corresponding to indices 0,1,2,3
            # Index mapping: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            U[0][col] += pat['U'][0] # a
            U[1][col] += pat['U'][1] # b
            U[2][col] += pat['U'][2] # c
            U[3][col] += pat['U'][3] # d
            
            # V part: global index j*p + k_global
            # B is 2x16. j in 0..1, k in 0..15.
            # Block q covers columns 2q and 2q+1.
            # pat['V'] is [e, f, g, h] corresponding to local cols 0,1 for rows 0,1.
            # Local indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            # Global k for local 0 is 2q, local 1 is 2q+1.
            
            k0 = 2 * q
            k1 = 2 * q + 1
            
            V[0 * p + k0][col] += pat['V'][0] # e (row 0, col 0)
            V[0 * p + k1][col] += pat['V'][1] # f (row 0, col 1)
            V[1 * p + k0][col] += pat['V'][2] # g (row 1, col 0)
            V[1 * p + k1][col] += pat['V'][3] # h (row 1, col 1)
            
            # W part: global index k_global * n + i
            # C is 2x16. k in 0..15, i in 0..1.
            # pat['W'] is [i, j, k, l] corresponding to local cols 0,1 for rows 0,1.
            # Local indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            # Note: The output C is 2x16.
            # The trace convention is W[k*n + i, r].
            # So index is k_global * 2 + i.
            
            W[k0 * n + 0][col] += pat['W'][0] # i (row 0, col 0)
            W[k0 * n + 1][col] += pat['W'][1] # j (row 0, col 1)
            W[k1 * n + 0][col] += pat['W'][2] # k (row 1, col 0)
            W[k1 * n + 1][col] += pat['W'][3] # l (row 1, col 1)
            
    # Convert to lists to match interface if required, or return arrays if allowed.
    # The interface says "factor matrices of shapes... with integer/half-integer entries".
    # It doesn't strictly say lists, but the seed used lists.
    # I will return lists to be safe.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
