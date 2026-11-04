import time
import random
from itertools import product

DIMENSIONS = [4, 4, 9]
DEADLINE = 135.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    def check_time():
        return time.time() - t0 < DEADLINE
    
    def validate(U, V, W, r):
        """Validate decomposition. U, V, W are lists of lists of ints/floats."""
        for i in range(n):
            for j in range(m):
                Uj = U[i * m + j]
                for k in range(p):
                    Vk = V[j * p + k]
                    Wk = W[k * n + i]
                    tot = 0
                    for c in range(r):
                        tot += Uj[c] * Vk[c] * Wk[c]
                    if tot != 1:
                        return False
        return True
    
    # Strassen 2x2x2: 7 terms
    STRASSEN = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 0, 1, 0]),   # M2
        ([0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]),    # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 0]),   # M4
        ([1, 1, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]),    # M5
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]),   # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]),   # M7
    ]
    
    def build_4x4x4_strassen():
        """Build rank-49 decomposition of 4x4x4 multiplication tensor.
        Returns list of (a_idx, b_idx, c_idx, coeff) tuples."""
        terms = []
        for outer in range(7):
            oa, ob, oc = STRASSEN[outer]
            for inner in range(7):
                ia, ib, ic = STRASSEN[inner]
                for cbi in range(4):
                    outer_c = oc[cbi]
                    if outer_c == 0:
                        continue
                    ci, cj = divmod(cbi, 2)  # block row, block col
                    for inner_cbi in range(4):
                        inner_c = ic[inner_cbi]
                        if inner_c == 0:
                            continue
                        inner_ci, inner_cj = divmod(inner_cbi, 2)
                        c_row = ci * 2 + inner_ci
                        c_col = cj * 2 + inner_cj
                        c_global = c_row * 4 + c_col
                        
                        for li in range(2):
                            for lj in range(2):
                                ia_c = ia[li * 2 + lj]
                                if ia_c == 0:
                                    continue
                                for ri in range(2):
                                    for rj in range(2):
                                        ib_c = ib[ri * 2 + rj]
                                        if ib_c == 0:
                                            continue
                                        for ab in range(4):
                                            ac = oa[ab]
                                            if ac == 0:
                                                continue
                                            abr, abc = divmod(ab, 2)
                                            a_row = 2 * abr + li
                                            a_col = 2 * abc + lj
                                            a_global = a_row * 4 + a_col
                                            
                                            for bb in range(4):
                                                bc = ob[bb]
                                                if bc == 0:
                                                    continue
                                                bbr, bbc = divmod(bb, 2)
                                                b_row = 2 * bbr + ri
                                                b_col = 2 * bbc + rj
                                                b_global = b_row * 4 + b_col
                                                
                                                coeff = outer_c * inner_c * ac * ia_c * ib_c * bc
                                                if coeff != 0:
                                                    terms.append((a_global, b_global, c_global, coeff))
        
        # Merge terms with same (a, b, c)
        term_dict = {}
        for t in terms:
            key = (t[0], t[1], t[2])
            if key in term_dict:
                term_dict[key] += t[3]
            else:
                term_dict[key] = t[3]
        
        return [(a, b, c, coeff) for (a, b, c), coeff in term_dict.items() if coeff != 0]
    
    def build_4x4x5_schoolbook():
        """Build schoolbook decomposition for 4x4x5 (k=4..8)."""
        terms = []
        for i in range(4):
            for j in range(4):
                for k in range(4, 9):
                    a_global = i * 4 + j
                    b_global = j * 9 + k
                    c_global = k * 4 + i
                    terms.append((a_global, b_global, c_global, 1))
        return terms
    
    # Build initial decomposition
    terms_4x4 = build_4x4x4_strassen()
    terms_4x5 = build_4x4x5_schoolbook()
    all_terms = terms_4x4 + terms_4x5
    
    r = len(all_terms)
    U = [[0] * r for _ in range(nm)]
    V = [[0] * r for _ in range(mp)]
    W = [[0] * r for _ in range(pn)]
    
    for c, (a, b, cg, coeff) in enumerate(all_terms):
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = cg // n
        i2 = cg % n
        U[i * m + j][c] += coeff
        V[j * p + k][c] += 1
        W[k * n + i][c] += 1
    
    if not validate(U, V, W, r):
        # Fallback: pure schoolbook
        r_fb = n * m * p
        U = [[0] * r_fb for _ in range(nm)]
        V = [[0] * r_fb for _ in range(mp)]
        W = [[0] * r_fb for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W, best_r = U[:], V[:], W[:], r
    
    # Fast validation using precomputed structure
    # For each (i,j,k), we need sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1
    # Precompute which columns contribute to which (i,j,k)
    
    def fast_validate(U, V, W, r):
        """Fast validation using sparse structure."""
        # For each (i,j,k), compute the sum
        for i in range(n):
            for j in range(m):
                Uj = U[i * m + j]
                for k in range(p):
                    Vk = V[j * p + k]
                    Wk = W[k * n + i]
                    tot = 0.0
                    for c in range(r):
                        uc = Uj[c]
                        if uc == 0:
                            continue
                        vc = Vk[c]
                        if vc == 0:
                            continue
                        wc = Wk[c]
                        if wc == 0:
                            continue
                        tot += uc * vc * wc
                    if tot != 1.0:
                        return False
        return True
    
    def try_eliminate(U, V, W, r, col):
        r_new = r - 1
        if r_new <= 0:
            return None
        U_new = [row[:col] + row[col+1:] for row in U]
        V_new = [row[:col] + row[col+1:] for row in V]
        W_new = [row[:col] + row[col+1:] for row in W]
        if fast_validate(U_new, V_new, W_new, r_new):
            return U_new, V_new, W_new, r_new
        return None
    
    # Greedy elimination with time budget
    improved = True
    while improved and check_time():
        improved = False
        for col in range(r):
            if not check_time():
                break
            result = try_eliminate(U, V, W, r, col)
            if result is not None:
                U, V, W, r = result
                improved = True
                if r < best_r:
                    best_U, best_V, best_W, best_r = U[:], V[:], W[:], r
                break
    
    # Try pair elimination with time budget
    if check_time():
        for c1 in range(r):
            if not check_time():
                break
            for c2 in range(c1 + 1, r):
                if not check_time():
                    break
                U_t = [row[:c1] + row[c1+1:c2] + row[c2+1:] for row in U]
                V_t = [row[:c1] + row[c1+1:c2] + row[c2+1:] for row in V]
                W_t = [row[:c1] + row[c1+1:c2] + row[c2+1:] for row in W]
                if fast_validate(U_t, V_t, W_t, r - 2):
                    U, V, W, r = U_t, V_t, W_t, r - 2
                    if r < best_r:
                        best_U, best_V, best_W, best_r = U[:], V[:], W[:], r
                    break
    
    # Try triple elimination with time budget
    if check_time() and r > 3:
        for c1 in range(r):
            if not check_time():
                break
            for c2 in range(c1 + 1, r):
                if not check_time():
                    break
                for c3 in range(c2 + 1, r):
                    if not check_time():
                        break
                    U_t = [row[:c1] + row[c1+1:c2] + row[c2+1:c3] + row[c3+1:] for row in U]
                    V_t = [row[:c1] + row[c1+1:c2] + row[c2+1:c3] + row[c3+1:] for row in V]
                    W_t = [row[:c1] + row[c1+1:c2] + row[c2+1:c3] + row[c3+1:] for row in W]
                    if fast_validate(U_t, V_t, W_t, r - 3):
                        U, V, W, r = U_t, V_t, W_t, r - 3
                        if r < best_r:
                            best_U, best_V, best_W, best_r = U[:], V[:], W[:], r
                        break
    
    return {"U": best_U, "V": best_V, "W": best_W}
