import sys
from functools import lru_cache

# Increase recursion limit just in case, though dims are small
sys.setrecursionlimit(2000)

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Note: Task says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # My variables: m (cols of A / rows of B), n (rows of A), p (cols of B)
    # Task dims: [3, 4, 6] -> n=3, m=4, p=6.
    
    terms = get_terms(n, m, p)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][r] = val
            
    return {"U": U, "V": V, "W": W}

@lru_cache(None)
def get_best_rank(m, n, p):
    # m: cols of A / rows of B
    # n: rows of A
    # p: cols of B
    
    # Base cases
    if m == 1 or n == 1 or p == 1:
        return m * n * p
    if m == 2 and n == 2 and p == 2:
        return 7
        
    best = m * n * p
    
    # Try splitting n (rows of A)
    for n1 in range(1, n):
        n2 = n - n1
        r = get_best_rank(m, n1, p) + get_best_rank(m, n2, p)
        if r < best:
            best = r
            
    # Try splitting m (cols of A)
    for m1 in range(1, m):
        m2 = m - m1
        r = get_best_rank(m1, n, p) + get_best_rank(m2, n, p)
        if r < best:
            best = r
            
    # Try splitting p (cols of B)
    for p1 in range(1, p):
        p2 = p - p1
        r = get_best_rank(m, n, p1) + get_best_rank(m, n, p2)
        if r < best:
            best = r
            
    return best

def get_terms(m, n, p):
    # Returns list of (u_dict, v_dict, w_dict)
    
    # Base cases
    if m == 1 or n == 1 or p == 1:
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Term for A[i,j] * B[j,k]
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        return terms
        
    if m == 2 and n == 2 and p == 2:
        return strassen_terms()
        
    # Find best split
    best_rank = m * n * p
    best_split = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        r = get_best_rank(m, n1, p) + get_best_rank(m, n2, p)
        if r < best_rank:
            best_rank = r
            best_split = ('n', n1, n2)
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        r = get_best_rank(m1, n, p) + get_best_rank(m2, n, p)
        if r < best_rank:
            best_rank = r
            best_split = ('m', m1, m2)
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        r = get_best_rank(m, n, p1) + get_best_rank(m, n, p2)
        if r < best_rank:
            best_rank = r
            best_split = ('p', p1, p2)
            
    if best_split is None:
        # Fallback to schoolbook (should not happen if base cases cover it)
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        return terms
        
    dim, d1, d2 = best_split
    
    if dim == 'n':
        # Split n into n1, n2
        # A = [A1, A2], B = [B1; B2]
        # C = A1 B1 + A2 B2
        terms1 = get_terms(m, d1, p)
        terms2 = get_terms(m, d2, p)
        
        final_terms = []
        # Add terms1 (indices i in [0, d1), j in [0, m))
        for u, v, w in terms1:
            final_terms.append((u, v, w))
            
        # Add terms2 (indices i in [d1, d1+d2), j in [0, m))
        # Shift i by d1
        for u, v, w in terms2:
            u_shifted = {}
            for (i, j), val in u.items():
                u_shifted[(i + d1, j)] = val
            # v, w unchanged?
            # v: (j, k). j is col of A. Unchanged.
            # w: (k, i). i is row of C. Shifted.
            w_shifted = {}
            for (k, i), val in w.items():
                w_shifted[(k, i + d1)] = val
            final_terms.append((u_shifted, v, w_shifted))
            
        return final_terms
        
    elif dim == 'm':
        # Split m into m1, m2
        # A = [A1, A2], B = [B1; B2]
        # C = A1 B1 + A2 B2
        terms1 = get_terms(d1, n, p)
        terms2 = get_terms(d2, n, p)
        
        final_terms = []
        # Add terms1 (j in [0, d1))
        for u, v, w in terms1:
            final_terms.append((u, v, w))
            
        # Add terms2 (j in [d1, d1+d2))
        for u, v, w in terms2:
            u_shifted = {}
            for (i, j), val in u.items():
                u_shifted[(i, j + d1)] = val
            v_shifted = {}
            for (j, k), val in v.items():
                v_shifted[(j + d1, k)] = val
            # w unchanged
            final_terms.append((u_shifted, v_shifted, w))
            
        return final_terms
        
    else: # dim == 'p'
        # Split p into p1, p2
        # B = [B1, B2]
        # C = [C1, C2]
        terms1 = get_terms(m, n, d1)
        terms2 = get_terms(m, n, d2)
        
        final_terms = []
        # Add terms1 (k in [0, d1))
        for u, v, w in terms1:
            final_terms.append((u, v, w))
            
        # Add terms2 (k in [d1, d1+d2))
        for u, v, w in terms2:
            # u unchanged
            v_shifted = {}
            for (j, k), val in v.items():
                v_shifted[(j, k + d1)] = val
            w_shifted = {}
            for (k, i), val in w.items():
                w_shifted[(k + d1, i)] = val
            final_terms.append((u, v_shifted, w_shifted))
            
        return final_terms

def strassen_terms():
    # Returns list of (u_dict, v_dict, w_dict) for 2x2x2
    # Indices: i, j, k in {0, 1}
    # U: (i, j), V: (j, k), W: (k, i)
    
    terms = []
    
    # m1 = (a+d)(e+h)
    u = {(0,0): 1, (1,1): 1}
    v = {(0,0): 1, (1,1): 1}
    w = {(0,0): 1, (1,1): 1} # p += 1, s += 1
    terms.append((u, v, w))
    
    # m2 = (c+d)e
    u = {(1,0): 1, (1,1): 1}
    v = {(0,0): 1}
    w = {(0,1): 1, (1,1): -1} # r += 1, s -= 1
    terms.append((u, v, w))
    
    # m3 = a(f-h)
    u = {(0,0): 1}
    v = {(0,1): 1, (1,1): -1}
    w = {(1,0): 1, (1,1): 1} # q += 1, s += 1
    terms.append((u, v, w))
    
    # m4 = d(g-c)
    u = {(1,1): 1}
    v = {(1,0): 1, (0,0): -1}
    w = {(0,0): 1, (0,1): 1} # p += 1, r += 1
    terms.append((u, v, w))
    
    # m5 = (a+b)h
    u = {(0,0): 1, (0,1): 1}
    v = {(1,1): 1}
    w = {(0,0): -1, (1,0): 1} # p -= 1, q += 1
    terms.append((u, v, w))
    
    # m6 = (c-d)(f+h)
    u = {(1,0): 1, (1,1): -1}
    v = {(0,1): 1, (1,1): 1}
    w = {(1,1): 1} # s += 1
    terms.append((u, v, w))
    
    # m7 = (b-c)(e+g)
    u = {(0,1): 1, (1,0): -1}
    v = {(0,0): 1, (1,0): 1}
    w = {(0,0): 1} # p += 1
    terms.append((u, v, w))
    
    return terms
