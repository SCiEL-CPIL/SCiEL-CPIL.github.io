"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strassen 2x2x2 decomposition
    # U: (4,7) for A[i,j], V: (4,7) for B[j,k], W: (4,7) for C with W[k*2+i]
    sU = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    sV = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    sW = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])

    # Outer decomposition: 2x2 x 2x3 -> 2x3 block multiplication
    # 7 Strassen terms for cols 0,1 + 4 schoolbook terms for col 2
    outer_terms = []

    for t in range(7):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for ai in range(2):
            for bj in range(2):
                coeff = sU[ai*2+bj, t]
                if coeff != 0:
                    U_dict[(ai,bj)] = U_dict.get((ai,bj), 0) + coeff
        for bj in range(2):
            for ck in range(2):
                coeff = sV[bj*2+ck, t]
                if coeff != 0:
                    V_dict[(bj,ck)] = V_dict.get((bj,ck), 0) + coeff
        for ai in range(2):
            for ck in range(2):
                coeff = sW[ck*2+ai, t]
                if coeff != 0:
                    W_dict[(ai,ck)] = W_dict.get((ai,ck), 0) + coeff
        outer_terms.append({'U': U_dict, 'V': V_dict, 'W': W_dict})

    for ai in range(2):
        for bj in range(2):
            outer_terms.append({
                'U': {(ai,bj): 1},
                'V': {(bj,2): 1},
                'W': {(ai,2): 1}
            })

    total_rank = len(outer_terms) * 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)

    for ot_idx, outer in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t

            for (ai, bj), coeff_outer in outer['U'].items():
                for a in range(2):
                    for b in range(2):
                        coeff_inner = sU[a*2+b, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            u_row = (2*ai + a) * 4 + (2*bj + b)
                            U[u_row, col] += coeff_outer * coeff_inner

            for (bj, ck), coeff_outer in outer['V'].items():
                for b in range(2):
                    for c in range(2):
                        coeff_inner = sV[b*2+c, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            v_row = (2*bj + b) * 6 + (2*ck + c)
                            V[v_row, col] += coeff_outer * coeff_inner

            for (ai, ck), coeff_outer in outer['W'].items():
                for a in range(2):
                    for c in range(2):
                        coeff_inner = sW[c*2+a, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            w_row = (2*ck + c) * 4 + (2*ai + a)
                            W[w_row, col] += coeff_outer * coeff_inner

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
