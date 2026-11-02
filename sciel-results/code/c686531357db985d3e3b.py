import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    S = 7
    total_rank = S * S

    # Strassen factors for 2x2: indices (a,b) in 0..3 -> row, r in 0..6 -> col
    # A layout: a00=0, a01=1, a10=2, a11=3
    StrassenU = [
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, -1],
    ]
    StrassenV = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, -1, -1, 0, 1, 0, 1],
    ]
    StrassenW = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]

    U = [[0] * total_rank for _ in range(n * m)]   # 8 x 49
    V = [[0] * total_rank for _ in range(m * p)]   # 32 x 49
    W = [[0] * total_rank for _ in range(p * n)]   # 16 x 49

    for ro in range(S):
        for ri in range(S):
            rk = ro * S + ri

            # U contribution: outer[ao][ro] * inner[jo][ri]
            for ao in range(2):
                uo = StrassenU[ao][ro]
                if uo == 0:
                    continue
                for jo in range(2):
                    ui = StrassenU[jo][ri]
                    if ui == 0:
                        continue
                    U[ao * m + jo * 2 + jo][rk] += uo * ui

            # V contribution: outer[ko][ro] * inner[ki][ri]
            # V row = k_global * p + j_global, j_global = jo*2 + ki
            for ko in range(2):
                vo = StrassenV[ko][ro]
                if vo == 0:
                    continue
                for ki in range(2):
                    vi = StrassenV[ki][ri]
                    if vi == 0:
                        continue
                    V[(ko * 2 + ki) * p + ko * 2 + ki][rk] += vo * vi

            # W contribution: outer[co][ro] * inner[ci][ri]
            # W index = k_global * n + i_global, k_global = ko*2+ki, i_global = co
            for co in range(2):
                wo = StrassenW[co][ro]
                if wo == 0:
                    continue
                for ci in range(2):
                    wi = StrassenW[ci][ri]
                    if wi == 0:
                        continue
                    W[(co * 2 + ci) * n + co][rk] += wo * wi

    if time.time() - t0 > 118:
        pass

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
