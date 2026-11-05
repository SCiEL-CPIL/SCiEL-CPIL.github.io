import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    terms = []  # Each term: (U_dict, V_dict, W_dict)

    def add_term(u_dict, v_dict, w_dict):
        if time.time() > deadline:
            return
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))

    def add_strassen_2x2(r1, c1, k1):
        """Add 7 Strassen terms for a 2x2x2 block at (r1,c1,k1)."""
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        
        C00 = k1 * n + r1
        C01 = (k1 + 1) * n + r1
        C10 = k1 * n + (r1 + 1)
        C11 = (k1 + 1) * n + (r1 + 1)
        
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1})

    def add_block(r1, c1, k1, ar, ac, bc):
        """Add optimal-rank terms for a block of size ar x ac x bc."""
        # U index: (r1+i)*m + (c1+j)
        # V index: (c1+j)*p + (k1+k)
        # W index: (k1+k)*n + (r1+i)
        
        if ar >= 2 and ac >= 2 and bc >= 2:
            # Use Strassen only for exactly 2x2x2
            if ar == 2 and ac == 2 and bc == 2:
                add_strassen_2x2(r1, c1, k1)
            else:
                # Fallback to schoolbook
                for ii in range(ar):
                    for jj in range(ac):
                        for kk in range(bc):
                            u_idx = (r1 + ii) * m + (c1 + jj)
                            v_idx = (c1 + jj) * p + (k1 + kk)
                            w_idx = (k1 + kk) * n + (r1 + ii)
                            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        elif ar == 1 and ac == 1 and bc == 1:
            u_idx = r1 * m + c1
            v_idx = c1 * p + k1
            w_idx = k1 * n + r1
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        elif ar == 1 and ac == 1 and bc == 2:
            # C[0,k] = A[0,0] * B[0,k], k in {0,1}
            u_idx = r1 * m + c1
            v0 = c1 * p + k1
            v1 = c1 * p + (k1 + 1)
            w0 = k1 * n + r1
            w1 = (k1 + 1) * n + r1
            add_term({u_idx: 1}, {v0: 1, v1: 1}, {w0: 1, w1: 1})
        elif ar == 1 and ac == 1 and bc == 3:
            for kk in range(bc):
                u_idx = r1 * m + c1
                v_idx = c1 * p + (k1 + kk)
                w_idx = (k1 + kk) * n + r1
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        elif ar == 1 and ac == 2 and bc == 1:
            # C[0,0] = sum_j A[0,j]*B[j,0]
            u0 = r1 * m + c1
            u1 = r1 * m + (c1 + 1)
            v0 = c1 * p + k1
            v1 = (c1 + 1) * p + k1
            w0 = k1 * n + r1
            add_term({u0: 1, u1: 1}, {v0: 1, v1: 1}, {w0: 1})
        elif ar == 1 and ac == 3 and bc == 1:
            u_dict = {}
            v_dict = {}
            for jj in range(ac):
                u_dict[r1 * m + (c1 + jj)] = 1
                v_dict[(c1 + jj) * p + k1] = 1
            w0 = k1 * n + r1
            add_term(u_dict, v_dict, {w0: 1})
        elif ar == 1 and ac == 2 and bc == 2:
            # C[0,k] = sum_j A[0,j]*B[j,k], rank 2
            for jj in range(ac):
                u_idx = r1 * m + (c1 + jj)
                v0 = (c1 + jj) * p + k1
                v1 = (c1 + jj) * p + (k1 + 1)
                w0 = k1 * n + r1
                w1 = (k1 + 1) * n + r1
                add_term({u_idx: 1}, {v0: 1, v1: 1}, {w0: 1, w1: 1})
        elif ar == 1 and ac == 2 and bc == 3:
            for jj in range(ac):
                u_idx = r1 * m + (c1 + jj)
                v_dict = {}
                w_dict = {}
                for kk in range(bc):
                    v_dict[(c1 + jj) * p + (k1 + kk)] = 1
                    w_dict[(k1 + kk) * n + r1] = 1
                add_term({u_idx: 1}, v_dict, w_dict)
        elif ar == 1 and ac == 3 and bc == 2:
            for jj in range(ac):
                u_idx = r1 * m + (c1 + jj)
                v0 = (c1 + jj) * p + k1
                v1 = (c1 + jj) * p + (k1 + 1)
                w0 = k1 * n + r1
                w1 = (k1 + 1) * n + r1
                add_term({u_idx: 1}, {v0: 1, v1: 1}, {w0: 1, w1: 1})
        elif ar == 1 and ac == 3 and bc == 3:
            for jj in range(ac):
                u_idx = r1 * m + (c1 + jj)
                v_dict = {}
                w_dict = {}
                for kk in range(bc):
                    v_dict[(c1 + jj) * p + (k1 + kk)] = 1
                    w_dict[(k1 + kk) * n + r1] = 1
                add_term({u_idx: 1}, v_dict, w_dict)
        elif ar == 2 and ac == 1 and bc == 1:
            # C[i,0] = A[i,0]*B[0,0], rank 1
            u0 = r1 * m + c1
            u1 = (r1 + 1) * m + c1
            v0 = c1 * p + k1
            w0 = k1 * n + r1
            w1 = k1 * n + (r1 + 1)
            add_term({u0: 1, u1: 1}, {v0: 1}, {w0: 1, w1: 1})
        elif ar == 3 and ac == 1 and bc == 1:
            u_dict = {}
            w_dict = {}
            for ii in range(ar):
                u_dict[(r1 + ii) * m + c1] = 1
                w_dict[k1 * n + (r1 + ii)] = 1
            v0 = c1 * p + k1
            add_term(u_dict, {v0: 1}, w_dict)
        elif ar == 2 and ac == 1 and bc == 2:
            # C[i,k] = A[i,0]*B[0,k], rank 2
            for ii in range(ar):
                u_idx = (r1 + ii) * m + c1
                v0 = c1 * p + k1
                v1 = c1 * p + (k1 + 1)
                w0 = k1 * n + (r1 + ii)
                w1 = (k1 + 1) * n + (r1 + ii)
                add_term({u_idx: 1}, {v0: 1, v1: 1}, {w0: 1, w1: 1})
        elif ar == 3 and ac == 1 and bc == 2:
            for ii in range(ar):
                u_idx = (r1 + ii) * m + c1
                v0 = c1 * p + k1
                v1 = c1 * p + (k1 + 1)
                w0 = k1 * n + (r1 + ii)
                w1 = (k1 + 1) * n + (r1 + ii)
                add_term({u_idx: 1}, {v0: 1, v1: 1}, {w0: 1, w1: 1})
        elif ar == 2 and ac == 1 and bc == 3:
            for ii in range(ar):
                u_idx = (r1 + ii) * m + c1
                v_dict = {}
                w_dict = {}
                for kk in range(bc):
                    v_dict[c1 * p + (k1 + kk)] = 1
                    w_dict[(k1 + kk) * n + (r1 + ii)] = 1
                add_term({u_idx: 1}, v_dict, w_dict)
        elif ar == 3 and ac == 1 and bc == 3:
            for ii in range(ar):
                u_idx = (r1 + ii) * m + c1
                v_dict = {}
                w_dict = {}
                for kk in range(bc):
                    v_dict[c1 * p + (k1 + kk)] = 1
                    w_dict[(k1 + kk) * n + (r1 + ii)] = 1
                add_term({u_idx: 1}, v_dict, w_dict)
        elif ar == 2 and ac == 2 and bc == 1:
            # C[i,0] = sum_j A[i,j]*B[j,0], rank 2
            for jj in range(ac):
                u0 = r1 * m + (c1 + jj)
                u1 = (r1 + 1) * m + (c1 + jj)
                v0 = (c1 + jj) * p + k1
                w0 = k1 * n + r1
                w1 = k1 * n + (r1 + 1)
                add_term({u0: 1, u1: 1}, {v0: 1}, {w0: 1, w1: 1})
        elif ar == 3 and ac == 2 and bc == 1:
            for jj in range(ac):
                u_dict = {}
                w_dict = {}
                for ii in range(ar):
                    u_dict[(r1 + ii) * m + (c1 + jj)] = 1
                    w_dict[k1 * n + (r1 + ii)] = 1
                v0 = (c1 + jj) * p + k1
                add_term(u_dict, {v0: 1}, w_dict)
        elif ar == 2 and ac == 3 and bc == 1:
            for jj in range(ac):
                u0 = r1 * m + (c1 + jj)
                u1 = (r1 + 1) * m + (c1 + jj)
                v0 = (c1 + jj) * p + k1
                w0 = k1 * n + r1
                w1 = k1 * n + (r1 + 1)
                add_term({u0: 1, u1: 1}, {v0: 1}, {w0: 1, w1: 1})
        elif ar == 3 and ac == 3 and bc == 1:
            for jj in range(ac):
                u_dict = {}
                w_dict = {}
                for ii in range(ar):
                    u_dict[(r1 + ii) * m + (c1 + jj)] = 1
                    w_dict[k1 * n + (r1 + ii)] = 1
                v0 = (c1 + jj) * p + k1
                add_term(u_dict, {v0: 1}, w_dict)
        else:
            # Fallback schoolbook
            for ii in range(ar):
                for jj in range(ac):
                    for kk in range(bc):
                        u_idx = (r1 + ii) * m + (c1 + jj)
                        v_idx = (c1 + jj) * p + (k1 + kk)
                        w_idx = (k1 + kk) * n + (r1 + ii)
                        add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    blocks = [(0, 2), (2, 4), (4, 5)]

    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                add_block(r1, c1, k1, ar, ac, bc)

    # Remove duplicate terms
    seen = set()
    unique_terms = []
    for t in terms:
        u_dict, v_dict, w_dict = t
        key = (tuple(sorted(u_dict.items())), tuple(sorted(v_dict.items())), tuple(sorted(w_dict.items())))
        if key not in seen:
            seen.add(key)
            unique_terms.append(t)
    terms = unique_terms

    rank = len(terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}
