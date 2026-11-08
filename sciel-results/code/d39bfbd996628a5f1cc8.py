"""Matrix multiplication 3x4x8 rank-88 exact tensor decomposition."""
import time

_TIME_LIMIT = 145.0  # 5s margin from 150s limit

def construct():
    t0 = time.time()
    n, m, p = 3, 4, 8
    total_rank = 88  # 32 + 56

    u_rows = n * m  # 12
    v_rows = m * p  # 32
    w_rows = p * n  # 24

    U_flat = [0] * (u_rows * total_rank)
    V_flat = [0] * (v_rows * total_rank)
    W_flat = [0] * (w_rows * total_rank)

    # --- Block A: first row of A (i=0), full schoolbook 1x4x8 = 32 terms ---
    col = 0
    for jj in range(m):
        for kk in range(p):
            if time.time() - t0 > _TIME_LIMIT:
                break
            U_flat[(0 * m + jj) * total_rank + col] = 1
            V_flat[(jj * p + kk) * total_rank + col] = 1
            W_flat[(kk * n + 0) * total_rank + col] = 1
            col += 1
        if time.time() - t0 > _TIME_LIMIT:
            break

    if time.time() - t0 > _TIME_LIMIT:
        # Return partial or full? If we broke early, the rank is effectively higher/invalid if we don't fill it.
        # But the loop is tiny. This is just a safety net.
        pass

    # --- Block B: rows i=1,2; use 2x2 Strassen per (j_pair, k_pair) ---
    # Strassen 7-product factors for 2x2x2
    su = [
        (1, 0, 0, 1),   # M1
        (0, 0, 1, 1),   # M2
        (1, 0, 0, 0),   # M3
        (0, 0, 0, 1),   # M4
        (1, 1, 0, 0),   # M5
        (-1, 0, 1, 0),  # M6
        (0, 1, 0, -1)   # M7
    ]
    sv = [
        (1, 0, 0, 1),   # M1
        (1, 0, 0, 0),   # M2
        (0, 1, 0, -1),  # M3
        (-1, 0, 1, 0),  # M4
        (0, 0, 0, 1),   # M5
        (1, 1, 0, 0),   # M6
        (0, 0, 1, 1)    # M7
    ]
    sw = [
        (1, 0, 0, 1),   # M1
        (0, 0, 1, -1),  # M2
        (0, 1, 0, 1),   # M3
        (1, 0, 1, 0),   # M4
        (-1, 1, 0, 0),  # M5
        (0, 0, 0, 1),   # M6
        (1, 0, 0, 0)    # M7
    ]

    col_idx = 32
    for jb in range(m // 2):  # 2 blocks of j
        for kb in range(p // 2):  # 4 blocks of k
            for t in range(7):
                if time.time() - t0 > _TIME_LIMIT:
                    break
                
                # A-factors: local (li, lj) -> global row (1+li, jb*2+lj)
                for li in range(2):
                    for lj in range(2):
                        loc = li * 2 + lj
                        val = su[t][loc]
                        if val:
                            gi = 1 + li
                            gj = jb * 2 + lj
                            U_flat[(gi * m + gj) * total_rank + col_idx] += val

                # B-factors: local (lj, lk) -> global row (jb*2+lj, kb*2+lk)
                for lj in range(2):
                    for lk in range(2):
                        loc = lj * 2 + lk
                        val = sv[t][loc]
                        if val:
                            gj = jb * 2 + lj
                            gk = kb * 2 + lk
                            V_flat[(gj * p + gk) * total_rank + col_idx] += val

                # C-factors: local (li, lk) -> W index (gk*n + gi)
                for li in range(2):
                    for lk in range(2):
                        loc = li * 2 + lk
                        val = sw[t][loc]
                        if val:
                            gi = 1 + li
                            gk = kb * 2 + lk
                            W_flat[(gk * n + gi) * total_rank + col_idx] += val

                col_idx += 1
            if time.time() - t0 > _TIME_LIMIT:
                break
        if time.time() - t0 > _TIME_LIMIT:
            break

    # Reshape to (rows, rank)
    U_out = [U_flat[r * total_rank:(r + 1) * total_rank] for r in range(u_rows)]
    V_out = [V_flat[r * total_rank:(r + 1) * total_rank] for r in range(v_rows)]
    W_out = [W_flat[r * total_rank:(r + 1) * total_rank] for r in range(w_rows)]

    return {"U": U_out, "V": V_out, "W": W_out}
