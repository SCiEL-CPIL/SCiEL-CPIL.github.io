import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 7
    
    total_rank = 49 + 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Strassen 2x2x2 base terms
    s22 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    ]
    cm = {
        0: {(0,0): 1, (1,1): 1},
        1: {(1,0): 1, (1,1): -1},
        2: {(0,1): 1, (1,1): 1},
        3: {(0,0): 1, (1,0): 1},
        4: {(0,0): -1, (0,1): 1},
        5: {(1,1): 1},
        6: {(0,0): 1}
    }
    
    # Part 1: 4x4x4 (columns 0-3 of B)
    s44 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1})
    ]
    cblk = {
        0: {(0,0): 1, (1,1): 1},
        1: {(1,0): 1, (1,1): -1},
        2: {(0,1): 1, (1,1): 1},
        3: {(0,0): 1, (1,0): 1},
        4: {(0,0): -1, (0,1): 1},
        5: {(1,1): 1},
        6: {(0,0): 1}
    }
    
    for mi in range(7):
        xd, yd = s44[mi]
        for ti in range(7):
            r = current_rank
            current_rank += 1
            xs, ys = s22[ti]
            
            uc = [0.0] * 16
            vc = [0.0] * 28
            
            for (li, lj), cx in xs.items():
                for b, cb in xd.items():
                    bi, bj = b // 2, b % 2
                    i, j = 2*bi+li, 2*bj+lj
                    uc[i*4+j] += cx * cb
            
            for (li, lj), cy in ys.items():
                for b, cb in yd.items():
                    bi, bj = b // 2, b % 2
                    i, j = 2*bi+li, 2*bj+lj
                    vc[i*7+j] += cy * cb
            
            U[r-1] = uc
            V[r-1] = vc
            
            wc = [0.0] * 28
            for (li, lj), ccm in cm[ti].items():
                if ccm == 0: continue
                for (br, bc), cbcb in cblk[mi].items():
                    if cbcb == 0: continue
                    i_f, k_f = 2*br+li, 2*bc+lj
                    wc[k_f*4+i_f] += ccm * cbcb
            
            W[r-1] = wc
    
    # Part 2: 4x4x3 (columns 4-6 of B)
    def add_223(arb, acb, brb, crb):
        nonlocal current_rank
        
        for ti in range(7):
            r = current_rank
            current_rank += 1
            xs, ys = s22[ti]
            
            uc = [0.0] * 16
            vc = [0.0] * 28
            
            for (li, lj), cx in xs.items():
                i, j = 2*arb+li, 2*acb+lj
                uc[i*4+j] += cx
            
            for (li, lj), cy in ys.items():
                i, j = 2*brb+li, 4+lj
                vc[i*7+j] += cy
            
            U[r-1] = uc
            V[r-1] = vc
            
            wc = [0.0] * 28
            for (li, lj), ccm in cm[ti].items():
                i_f, k_f = 2*crb+li, 4+lj
                wc[k_f*4+i_f] += ccm
            
            W[r-1] = wc
        
        sb = [
            ({(0,0): 1}, {(0,0): 1}, (0,0)),
            ({(0,1): 1}, {(1,0): 1}, (0,0)),
            ({(1,0): 1}, {(0,0): 1}, (1,0)),
            ({(1,1): 1}, {(1,0): 1}, (1,0))
        ]
        for ti in range(4):
            r = current_rank
            current_rank += 1
            xs, ys, cp = sb[ti]
            
            uc = [0.0] * 16
            vc = [0.0] * 28
            
            for (li, lj), cx in xs.items():
                i, j = 2*arb+li, 2*acb+lj
                uc[i*4+j] += cx
            
            for (li, lj), cy in ys.items():
                i, j = 2*brb+li, 6
                vc[i*7+j] += cy
            
            U[r-1] = uc
            V[r-1] = vc
            
            wc = [0.0] * 28
            i_f, k_f = 2*crb+cp[0], 6
            wc[k_f*4+i_f] = 1.0
            
            W[r-1] = wc
    
    add_223(0, 0, 0, 0)
    add_223(0, 1, 1, 0)
    add_223(1, 0, 0, 1)
    add_223(1, 1, 1, 1)
    
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
