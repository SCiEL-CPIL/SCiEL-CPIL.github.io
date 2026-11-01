"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that no spurious terms exist
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _strassen_block(n, m, p):
    """
    Rank-56 Strassen-based decomposition for 2x2 * 2x16.
    Treats B as 8 independent 2x2 blocks, each multiplied by A using Strassen (rank 7).
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks of 2 columns each
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen intermediates for 2x2 * 2x2 -> 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C[0,0] = a*e + b*g, C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g, C[1,1] = c*f + d*h
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    strassen = [
        # (U_indices, V_indices, W_indices) for each M
        # M1 = (a+d)(e+h): U: a=idx0, d=idx3; V: e=col0, h=col1; W: C00=idx0, C11=idx3
        ([(0, 1), (3, 1)], [(0, 1), (1, 1)], [(0, 1), (3, 1)]),
        # M2 = (c+d)e: U: c=idx2, d=idx3; V: e=col0; W: C10=idx2, C11=idx3
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = a(f-h): U: a=idx0; V: f=col1, h=col1(offset); W: C01=idx1, C11=idx3
        ([(0, 1)], [(1, 1), (1, -1)], [(1, 1), (3, 1)]),
        # M4 = d(g-e): U: d=idx3; V: g=col2, e=col0; W: C00=idx0, C10=idx2
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (a+b)h: U: a=idx0, b=idx1; V: h=col1; W: C00=idx0, C01=idx1
        ([(0, 1), (1, 1)], [(1, 1)], [(0, -1), (1, 1)]),
        # M6 = (c-a)(e+g): U: c=idx2, a=idx0; V: e=col0, g=col2; W: C11=idx3
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),
        # M7 = (b-d)(f+h): U: b=idx1, d=idx3; V: f=col1, h=col1; W: C00=idx0
        ([(1, 1), (3, -1)], [(1, 1), (1, 1)], [(0, 1)]),
    ]
    
    col = 0
    for t in range(num_blocks):
        # For block t: B columns are 2t and 2t+1
        # V indices: row j, col k -> index j*16 + k
        # j=0: cols 2t, 2t+1 -> V indices 2t, 2t+1
        # j=1: cols 2t, 2t+1 -> V indices 16+2t, 16+2t+1
        # W indices: k*n+i = k*2+i
        # k=2t: W indices 4t, 4t+1
        # k=2t+1: W indices 4t+2, 4t+3
        
        # Map local B column indices to global V indices
        # local col 0 -> B[0, 2t] = V[2t]
        # local col 1 -> B[0, 2t+1] = V[2t+1]
        # local col 2 -> B[1, 2t] = V[16+2t]
        # local col 3 -> B[1, 2t+1] = V[16+2t+1]
        
        # But our Strassen uses: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        # So: e=V[2t], f=V[2t+1], g=V[16+2t], h=V[16+2t+1]
        
        v_map = {0: 2*t, 1: 2*t+1, 2: 16+2*t, 3: 16+2*t+1}
        w_map = {0: 4*t, 1: 4*t+1, 2: 4*t+2, 3: 4*t+3}
        
        for u_list, v_list, w_list in strassen:
            for u_idx, coeff in u_list:
                U[u_idx][col] += coeff
            for v_local, coeff in v_list:
                # v_local is the column index in the 2x2 B block
                # e=col0, f=col1, g=col2, h=col3
                v_global = v_map[v_local]
                V[v_global][col] += coeff
            for w_local, coeff in w_list:
                # w_local is the output position in 2x2 C block
                # C00=pos0, C01=pos1, C10=pos2, C11=pos3
                w_global = w_map[w_local]
                W[w_global][col] += coeff
            col += 1
    
    return U, V, W

def _rank48_direct(n, m, p):
    """
    Rank-48 decomposition: 16 k-values, each using rank-3 for 2x2x1.
    For each k: C[2k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
               C[2k+1] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    Rank-3 for 2x2x1:
    M1 = (A[0,0]+A[1,0]) * B[0,k]
    M2 = A[1,0] * B[1,k]
    M3 = A[0,1] * B[1,k]
    C[2k] = M1 - M2 + M3  = (A[0,0]+A[1,0])B[0,k] - A[1,0]B[1,k] + A[0,1]B[1,k]
           = A[0,0]B[0,k] + A[1,0]B[0,k] - A[1,0]B[1,k] + A[0,1]B[1,k]
    That's not right either. Let me use a correct rank-3.
    
    Correct rank-3 for computing [x;y] = [[a,b],[c,d]] * [p;q]:
    M1 = (a+c) * p
    M2 = c * (p+q) 
    M3 = b * q
    x = M1 - M2 + M3 = (a+c)p - c(p+q) + bq = ap + cp - cp - cq + bq = ap + bq ✓
    y = M1 + M2 - M3... = (a+c)p + c(p+q) - bq = ap + cp + cp + cq - bq = ap + 2cp + cq - bq ✗
    
    Let me try:
    M1 = a*p
    M2 = b*q
    M3 = (c+d)*p
    x = M1 + M2 ✓
    y = M3 - M1 + c*q... no, M3 only has p.
    
    OK the issue is that rank-3 for 2x2x1 requires:
    M1 = (a+d)(p+q)
    M2 = (c+d)p
    M3 = a(q-p)
    x = M1 - M2 + M3 = (a+d)(p+q) - (c+d)p + a(q-p)
       = ap + aq + dp + dq - cp - dp + aq - ap
       = 2aq + dq - cp
    That's not right.
    
    I'll just use rank-4 per k (schoolbook for 2x2x1) giving total rank 64,
    or try a different grouping.
    """
    # Fall back to strassen
    return _strassen_block(n, m, p)

def _rank40_grouped(n, m, p):
    """
    Try rank-40 by grouping 16 k-values into 8 pairs and using a smarter decomposition.
    For each pair (k=2t, k=2t+1), compute 4 outputs from A (2x2) and B[:, 2t:2t+2] (2x2).
    This is 2x2 * 2x2 -> 2x2, rank 7 per pair, total 8*7=56. Same as Strassen.
    
    Instead, let's try: group into 4 groups of 4 k-values.
    For each group of 4, we compute 8 outputs from A (2x2) and B[:, group] (2x4).
    This is 2x2 * 2x4 -> 2x4. Rank of this tensor?
    Schoolbook: 2*4 = 8 per group, total 4*8 = 32.
    
    Can we do better than 8 for 2x2*2x4?
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0,1,2,3.
    
    This is equivalent to: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    So we're computing 2 linear combinations of 2 vectors of length 4.
    
    Rank of 2x2x4 tensor: The tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k].
    This is a 2x2x4 tensor. Its rank is at most 2*4=8 (schoolbook).
    Lower bound: at least max(2,2,4)=4.
    
    Actually, this tensor is the same as the "matrix multiplication" tensor MM_{2,2,4} but transposed.
    The rank of MM_{2,2,p} for p>=2 is known to be 2p for p even (using Strassen-like).
    For p=4: rank = 8? Or can we do 7?
    
    MM_{2,2,2} has rank 7 (Strassen).
    MM_{2,2,4}: we can view as two independent MM_{2,2,2} if we split into pairs, giving rank 14.
    But schoolbook gives 8. So 8 is better.
    
    Can we do less than 8 for MM_{2,2,4}? 
    Lower bound: the tensor has 8 non-zero entries, so rank >= 8? No, rank can be less.
    
    Actually, I think rank(MM_{2,2,p}) = 2p for all p >= 2 based on known results.
    So for p=4, rank = 8. Total: 4 groups * 8 = 32.
    
    Let me implement rank-32 with 4 groups of 4 k-values each.
    """
    n, m, p = 2, 2, 16
    num_groups = 4
    k_per_group = p // num_groups  # 4
    rank = num_groups * 2 * k_per_group  # 4 * 2 * 4 = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for g in range(num_groups):
        # Group g handles k values: g*4, g*4+1, g*4+2, g*4+3
        for j in range(2):  # B row
            for k_local in range(k_per_group):
                k_global = g * k_per_group + k_local
                # U: A[i,j] for i=0,1 -> indices j (since a = i*2+j, so for fixed j, i=0->j, i=1->j+2)
                # V: B[j,k] -> index j*16 + k
                # W: C[i,k] -> index k*2 + i
                for i in range(2):
                    u_idx = i * 2 + j  # a = i*m + j = i*2 + j
                    v_idx = j * 16 + k_global
                    w_idx = k_global * 2 + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Try rank-32 grouped construction (4 groups of 4 k-values, schoolbook per group)
    try:
        U, V, W = _rank40_grouped(n, m, p)
        if time.time() < deadline and _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try Strassen-based construction (rank 56)
    try:
        U, V, W = _strassen_block(n, m, p)
        if time.time() < deadline and _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook (rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
