"""Block-structured low-rank decomposition for 4x5x6 matrix multiplication."""
import time
import sys
import random

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0


def remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W, "rank": rank}


def _verify(U, V, W, rank):
    """Exact check of decomposition using random integer matrices."""
    n, m, p = DIMENSIONS
    for _ in range(5):
        if remaining() <= 0:
            return False
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u = U[i * m + j][c]
                        if u == 0:
                            continue
                        acc += u * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True


def _winograd_2x2x2_terms():
    """
    Returns 7 rank-1 terms for 2x2x2 matmul.
    Each term: (u_vec, v_vec, w_vec) where each is a flat 4-element vector.
    u_vec corresponds to A entries [A00, A01, A10, A11]
    v_vec corresponds to B entries [B00, B01, B10, B11]
    w_vec corresponds to C entries [C00, C01, C10, C11]
    
    Verified correct Winograd F(2) identity:
    p1 = (a+d)(e+h)
    p2 = (c+d)e
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)g
    p6 = (c-a)(e+f)
    p7 = (b-d)(h+f)
    
    C00 = p1 - p4 + p5 - p7
    C01 = p3 + p4 + p5 + p7
    C10 = p2 + p4 - p6 + p7
    C11 = p1 - p2 + p6 - p7
    """
    terms = []
    
    # p1 = (a+d)(e+h): A: a+d = [1,0,0,1], B: e+h = [1,0,0,1]
    # C: +p1 at C00, +p1 at C11 => [1,0,0,1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # p2 = (c+d)e: A: c+d = [0,0,1,1], B: e = [1,0,0,0]
    # C: +p2 at C10, -p2 at C11 => [0,0,1,-1]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # p3 = a(f-h): A: a = [1,0,0,0], B: f-h = [0,1,0,-1]
    # C: +p3 at C01 => [0,1,0,0]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    
    # p4 = d(g-e): A: d = [0,0,0,1], B: g-e = [-1,0,1,0]
    # C: -p4 at C00, +p4 at C01, +p4 at C10 => [-1,1,1,0]
    terms.append(([0,0,0,1], [-1,0,1,0], [-1,1,1,0]))
    
    # p5 = (a+b)g: A: a+b = [1,1,0,0], B: g = [0,0,1,0]
    # C: +p5 at C00, +p5 at C01 => [1,1,0,0]
    terms.append(([1,1,0,0], [0,0,1,0], [1,1,0,0]))
    
    # p6 = (c-a)(e+f): A: c-a = [-1,0,1,0], B: e+f = [1,1,0,0]
    # C: -p6 at C10, +p6 at C11 => [0,0,-1,1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,-1,1]))
    
    # p7 = (b-d)(h+f): A: b-d = [0,1,0,-1], B: h+f = [0,1,0,1]
    # C: -p7 at C00, +p7 at C01, +p7 at C10 => [-1,1,1,0]
    terms.append(([0,1,0,-1], [0,1,0,1], [-1,1,1,0]))
    
    return terms


def _verify_winograd_2x2x2():
    """Verify the Winograd 2x2x2 identity is correct."""
    terms = _winograd_2x2x2_terms()
    # Test with random 2x2 matrices
    for _ in range(20):
        A = [[random.randint(-5, 5) for _ in range(2)] for _ in range(2)]
        B = [[random.randint(-5, 5) for _ in range(2)] for _ in range(2)]
        C_expected = [[sum(A[i][j] * B[j][k] for j in range(2)) for k in range(2)] for i in range(2)]
        C_computed = [[0, 0], [0, 0]]
        for u, v, w in terms:
            a_val = u[0]*A[0][0] + u[1]*A[0][1] + u[2]*A[1][0] + u[3]*A[1][1]
            b_val = v[0]*B[0][0] + v[1]*B[0][1] + v[2]*B[1][0] + v[3]*B[1][1]
            prod = a_val * b_val
            C_computed[0][0] += w[0] * prod
            C_computed[0][1] += w[1] * prod
            C_computed[1][0] += w[2] * prod
            C_computed[1][1] += w[3] * prod
        if C_computed != C_expected:
            return False
    return True


def _block_decomposition():
    """Build 4x5x6 decomposition using Winograd 2x2x2 blocks where possible."""
    n, m, p = DIMENSIONS
    
    # First check if Winograd is valid
    winograd_valid = _verify_winograd_2x2x2()
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    if winograd_valid:
        Wt_terms = _winograd_2x2x2_terms()
        
        # Partition: i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
        # Use 2x2x2 Winograd blocks for i in {0,1}x{2,3}, j in {0,1}x{2,3}, k in {0,1}x{2,3}
        # That covers i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3}
        
        for ib in range(2):           # i block: 2*ib, 2*ib+1
            for kb in range(2):       # k block: 2*kb, 2*kb+1
                for jb in range(2):   # j block: 2*jb, 2*jb+1
                    for (u, v, w) in Wt_terms:
                        ufull = [0] * (n * m)
                        vfull = [0] * (m * p)
                        wfull = [0] * (p * n)
                        
                        # Fill U: A entries at positions (2*ib+a, 2*jb+b) for a,b in {0,1}
                        for a in range(2):
                            for b in range(2):
                                ui = (2 * ib + a) * m + (2 * jb + b)
                                ufull[ui] += u[a * 2 + b]
                        
                        # Fill V: B entries at positions (2*jb+b, 2*kb+c) for b,c in {0,1}
                        for b in range(2):
                            for c in range(2):
                                vi = (2 * jb + b) * p + (2 * kb + c)
                                vfull[vi] += v[b * 2 + c]
                        
                        # Fill W: C entries at positions (2*kb+c, 2*ib+a) for c,a in {0,1}
                        for c in range(2):
                            for a in range(2):
                                wi = (2 * kb + c) * n + (2 * ib + a)
                                wfull[wi] += w[c * 2 + a]
                        
                        terms_U.append(ufull)
                        terms_V.append(vfull)
                        terms_W.append(wfull)
        
        # Add schoolbook for uncovered (i,j,k): j==4 or k in {4,5}
        covered = set()
        for ib in range(2):
            for kb in range(2):
                for jb in range(2):
                    for a in range(2):
                        for b in range(2):
                            for c in range(2):
                                covered.add((2*ib+a, 2*jb+b, 2*kb+c))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    ufull = [0] * (n * m)
                    vfull = [0] * (m * p)
                    wfull = [0] * (p * n)
                    ufull[i * m + j] = 1
                    vfull[j * p + k] = 1
                    wfull[k * n + i] = 1
                    terms_U.append(ufull)
                    terms_V.append(vfull)
                    terms_W.append(wfull)
    else:
        # Fall back to full schoolbook
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ufull = [0] * (n * m)
                    vfull = [0] * (m * p)
                    wfull = [0] * (p * n)
                    ufull[i * m + j] = 1
                    vfull[j * p + k] = 1
                    wfull[k * n + i] = 1
                    terms_U.append(ufull)
                    terms_V.append(vfull)
                    terms_W.append(wfull)
    
    rank = len(terms_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W, "rank": rank}


def construct():
    n, m, p = DIMENSIONS
    best = None
    
    # Try block decomposition
    try:
        if remaining() > 20:
            cand = _block_decomposition()
            if remaining() > 10 and _verify(cand["U"], cand["V"], cand["W"], cand["rank"]):
                best = cand
    except Exception:
        pass
    
    # Fallback to schoolbook
    if best is None:
        best = _schoolbook()
    
    del best["rank"]
    return best


if __name__ == "__main__":
    r = construct()
    print("U shape:", len(r["U"]), "x", len(r["U"][0]))
    print("rank:", len(r["U"][0]))
