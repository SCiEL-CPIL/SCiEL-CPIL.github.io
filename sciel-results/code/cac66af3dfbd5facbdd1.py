import time
import numpy as np

DEADLINE = 140.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60

    # Build candidate pool of rank-1 tensors
    candidates = []

    def add_candidate(u_sparse, v_sparse, w_sparse):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        for (i, j), val in u_sparse.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_sparse.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_sparse.items():
            w_vec[k * n + i] = val
        contrib = np.zeros(total_entries)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i * m * p + j * p + k] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        candidates.append((u_vec, v_vec, w_vec, contrib))

    # Type 1: Standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_candidate({(i,j): 1}, {(j,k): 1}, {(k,i): 1})

    # Type 2: Strassen 2x2x2 blocks
    def strassen_terms(io, jo, ko):
        terms = []
        terms.append(({(io,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1}, 
                      {(ko,io+1): 1, (ko+1,io+1): -1}))
        terms.append(({(io,jo): 1}, 
                      {(jo,ko+1): 1, (jo+1,ko+1): -1}, 
                      {(ko+1,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo+1): 1}, 
                      {(jo+1,ko): 1, (jo,ko): -1}, 
                      {(ko,io): 1, (ko,io+1): 1}))
        terms.append(({(io,jo): 1, (io,jo+1): 1}, 
                      {(jo+1,ko+1): 1}, 
                      {(ko,io): -1, (ko+1,io): 1}))
        terms.append(({(io+1,jo): 1, (io,jo): -1}, 
                      {(jo,ko): 1, (jo,ko+1): 1}, 
                      {(ko+1,io+1): 1}))
        terms.append(({(io,jo+1): 1, (io+1,jo+1): -1}, 
                      {(jo+1,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1}))
        return terms

    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)

    # Type 3: Wider blocks (2x3x2, 2x2x3, 2x3x3) with Strassen-like patterns
    # 2x3x2: use 3 Strassen blocks on pairs of j, plus corrections
    # Actually let's just add more diverse patterns

    # 2x3x2 blocks
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 1):
                # Use Strassen on (jo, jo+1) and (jo+1, jo+2), plus a term for jo
                # Term 1: standard (io,jo)
                add_candidate({(io,jo): 1}, {(jo,ko): 1}, {(ko,io): 1})
                add_candidate({(io+1,jo): 1}, {(jo,ko): 1}, {(ko,io+1): 1})
                # Strassen on (jo, jo+1)
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                # Strassen on (jo+1, jo+2)
                for u_c, v_c, w_c in strassen_terms(io, jo+1, ko):
                    add_candidate(u_c, v_c, w_c)
                # Additional cross terms for 2x3x2
                add_candidate({(io,jo+2): 1}, {(jo+2,ko): 1, (jo+2,ko+1): 1}, {(ko,io): 1, (ko+1,io): 1})
                add_candidate({(io+1,jo+2): 1}, {(jo+2,ko): 1, (jo+2,ko+1): 1}, {(ko,io+1): 1, (ko+1,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+1): 1, (jo+2,ko): 1}, {(ko,io): 1, (ko+1,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+1): 1, (jo+2,ko): -1}, {(ko+1,io): 1, (ko,io+1): 1})

    # 2x2x3 blocks
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 2):
                add_candidate({(io,jo): 1}, {(jo,ko): 1}, {(ko,io): 1})
                add_candidate({(io+1,jo): 1}, {(jo,ko): 1}, {(ko,io+1): 1})
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                for u_c, v_c, w_c in strassen_terms(io, jo, ko+1):
                    add_candidate(u_c, v_c, w_c)
                # Cross terms for 2x2x3
                add_candidate({(io,jo+1): 1}, {(jo+1,ko+2): 1, (jo,ko+2): 1}, {(ko+2,io): 1, (ko,io): 1})
                add_candidate({(io+1,jo+1): 1}, {(jo+1,ko+2): 1, (jo,ko+2): 1}, {(ko+2,io+1): 1, (ko,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+1): 1}, {(jo,ko+2): 1, (jo+1,ko+2): 1}, {(ko+2,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+1): 1}, {(jo,ko+2): 1, (jo+1,ko+2): -1}, {(ko+2,io+1): 1, (ko,io): 1})

    # 2x3x3 blocks - add diverse patterns
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                # Standard terms
                for j_off in range(3):
                    for k_off in range(3):
                        add_candidate({(io,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io): 1})
                        add_candidate({(io+1,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io+1): 1})
                # Strassen on sub-blocks
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                for u_c, v_c, w_c in strassen_terms(io, jo, ko+1):
                    add_candidate(u_c, v_c, w_c)
                for u_c, v_c, w_c in strassen_terms(io, jo+1, ko):
                    add_candidate(u_c, v_c, w_c)
                for u_c, v_c, w_c in strassen_terms(io, jo+1, ko+1):
                    add_candidate(u_c, v_c, w_c)
                # Cross terms for 2x3x3
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+2): 1, (jo+2,ko): 1}, {(ko,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+2): 1, (jo+2,ko): -1}, {(ko+2,io): 1, (ko,io+1): 1})
                add_candidate({(io,jo+2): 1, (io+1,jo): 1}, {(jo+2,ko): 1, (jo,ko+2): 1}, {(ko,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo+2): 1, (io+1,jo): 1}, {(jo+2,ko): 1, (jo,ko+2): -1}, {(ko+2,io): 1, (ko,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+1): 1}, {(jo+1,ko): 1, (jo+1,ko+2): 1}, {(ko,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+1): 1}, {(jo+1,ko): 1, (jo+1,ko+2): -1}, {(ko+2,io): 1, (ko,io+1): 1})

    # Type 4: Wider 2x4x2 and 2x2x4 blocks
    for io in range(n - 1):
        for jo in range(m - 3):
            for ko in range(p - 1):
                # Standard
                for j_off in range(4):
                    add_candidate({(io,jo+j_off): 1}, {(jo+j_off,ko): 1}, {(ko,io): 1})
                    add_candidate({(io+1,jo+j_off): 1}, {(jo+j_off,ko): 1}, {(ko,io+1): 1})
                # Strassen on pairs
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                for u_c, v_c, w_c in strassen_terms(io, jo+1, ko):
                    add_candidate(u_c, v_c, w_c)
                for u_c, v_c, w_c in strassen_terms(io, jo+2, ko):
                    add_candidate(u_c, v_c, w_c)
                # Cross terms
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+1): 1, (jo+3,ko): 1}, {(ko,io): 1, (ko+1,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+1): 1, (jo+3,ko): -1}, {(ko+1,io): 1, (ko,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+2): 1}, {(jo+1,ko+1): 1, (jo+2,ko): 1}, {(ko,io): 1, (ko+1,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+2): 1}, {(jo+1,ko+1): 1, (jo+2,ko): -1}, {(ko+1,io): 1, (ko,io+1): 1})

    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 3):
                # Standard
                for k_off in range(4):
                    add_candidate({(io,jo): 1}, {(jo,ko+k_off): 1}, {(ko+k_off,io): 1})
                    add_candidate({(io+1,jo): 1}, {(jo,ko+k_off): 1}, {(ko+k_off,io+1): 1})
                # Strassen on pairs
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                for u_c, v_c, w_c in strassen_terms(io, jo, ko+1):
                    add_candidate(u_c, v_c, w_c)
                for u_c, v_c, w_c in strassen_terms(io, jo, ko+2):
                    add_candidate(u_c, v_c, w_c)
                # Cross terms
                add_candidate({(io,jo+1): 1, (io+1,jo+1): 1}, {(jo+1,ko): 1, (jo+1,ko+3): 1}, {(ko,io): 1, (ko+3,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+1): 1}, {(jo+1,ko): 1, (jo+1,ko+3): -1}, {(ko+3,io): 1, (ko,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo): 1}, {(jo,ko+1): 1, (jo,ko+2): 1}, {(ko+1,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo): 1}, {(jo,ko+1): 1, (jo,ko+2): -1}, {(ko+2,io): 1, (ko+1,io+1): 1})

    # Type 5: 2x4x3, 2x3x4, 2x4x4 blocks - add diverse patterns
    for io in range(n - 1):
        for jo in range(m - 3):
            for ko in range(p - 2):
                for j_off in range(4):
                    for k_off in range(3):
                        add_candidate({(io,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io): 1})
                        add_candidate({(io+1,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io+1): 1})
                # Strassen on sub-blocks
                for j_sub in range(3):
                    for k_sub in range(2):
                        for u_c, v_c, w_c in strassen_terms(io, jo+j_sub, ko+k_sub):
                            add_candidate(u_c, v_c, w_c)
                # Cross terms
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+2): 1, (jo+3,ko): 1}, {(ko,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+2): 1, (jo+3,ko): -1}, {(ko+2,io): 1, (ko,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+2): 1}, {(jo+1,ko+2): 1, (jo+2,ko): 1}, {(ko,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+2): 1}, {(jo+1,ko+2): 1, (jo+2,ko): -1}, {(ko+2,io): 1, (ko,io+1): 1})

    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 3):
                for j_off in range(3):
                    for k_off in range(4):
                        add_candidate({(io,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io): 1})
                        add_candidate({(io+1,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io+1): 1})
                # Strassen on sub-blocks
                for j_sub in range(2):
                    for k_sub in range(3):
                        for u_c, v_c, w_c in strassen_terms(io, jo+j_sub, ko+k_sub):
                            add_candidate(u_c, v_c, w_c)
                # Cross terms
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+3): 1, (jo+2,ko): 1}, {(ko,io): 1, (ko+3,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+3): 1, (jo+2,ko): -1}, {(ko+3,io): 1, (ko,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+1): 1}, {(jo+1,ko+1): 1, (jo+1,ko+2): 1}, {(ko+1,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+1): 1}, {(jo+1,ko+1): 1, (jo+1,ko+2): -1}, {(ko+2,io): 1, (ko+1,io+1): 1})

    for io in range(n - 1):
        for jo in range(m - 3):
            for ko in range(p - 3):
                for j_off in range(4):
                    for k_off in range(4):
                        add_candidate({(io,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io): 1})
                        add_candidate({(io+1,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io+1): 1})
                # Strassen on sub-blocks
                for j_sub in range(3):
                    for k_sub in range(3):
                        for u_c, v_c, w_c in strassen_terms(io, jo+j_sub, ko+k_sub):
                            add_candidate(u_c, v_c, w_c)
                # Cross terms
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+3): 1, (jo+3,ko): 1}, {(ko,io): 1, (ko+3,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+3): 1, (jo+3,ko): -1}, {(ko+3,io): 1, (ko,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+2): 1}, {(jo+1,ko+3): 1, (jo+2,ko): 1}, {(ko,io): 1, (ko+3,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+2): 1}, {(jo+1,ko+3): 1, (jo+2,ko): -1}, {(ko+3,io): 1, (ko,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+2): 1}, {(jo+1,ko+1): 1, (jo+2,ko+2): 1}, {(ko+1,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+2): 1}, {(jo+1,ko+1): 1, (jo+2,ko+2): -1}, {(ko+2,io): 1, (ko+1,io+1): 1})

    # Type 6: Full 2x5x6 with diverse patterns
    # Add some global cross terms
    add_candidate({(0,0): 1, (1,4): 1}, {(0,5): 1, (4,0): 1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,0): 1, (1,4): 1}, {(0,5): 1, (4,0): -1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,1): 1, (1,3): 1}, {(1,4): 1, (3,1): 1}, {(1,0): 1, (4,1): 1})
    add_candidate({(0,1): 1, (1,3): 1}, {(1,4): 1, (3,1): -1}, {(4,0): 1, (1,1): 1})
    add_candidate({(0,2): 1, (1,2): 1}, {(2,2): 1, (2,3): 1}, {(2,0): 1, (3,1): 1})
    add_candidate({(0,2): 1, (1,2): 1}, {(2,2): 1, (2,3): -1}, {(3,0): 1, (2,1): 1})
    add_candidate({(0,0): 1, (1,1): 1, (1,4): 1}, {(0,5): 1, (1,0): 1, (4,1): 1}, {(0,0): 1, (1,1): 1, (5,1): 1})
    add_candidate({(0,0): 1, (1,1): 1, (1,4): 1}, {(0,5): 1, (1,0): -1, (4,1): -1}, {(5,0): 1, (0,1): 1, (1,1): 1})
    add_candidate({(0,3): 1, (1,3): 1}, {(3,2): 1, (3,3): 1}, {(2,0): 1, (3,1): 1})
    add_candidate({(0,3): 1, (1,3): 1}, {(3,2): 1, (3,3): -1}, {(3,0): 1, (2,1): 1})
    add_candidate({(0,4): 1, (1,4): 1}, {(4,0): 1, (4,1): 1}, {(0,0): 1, (1,1): 1})
    add_candidate({(0,4): 1, (1,4): 1}, {(4,0): 1, (4,1): -1}, {(1,0): 1, (0,1): 1})
    add_candidate({(0,0): 1, (1,0): 1}, {(0,2): 1, (0,3): 1}, {(2,0): 1, (3,1): 1})
    add_candidate({(0,0): 1, (1,0): 1}, {(0,2): 1, (0,3): -1}, {(3,0): 1, (2,1): 1})
    add_candidate({(0,4): 1, (1,4): 1}, {(4,4): 1, (4,5): 1}, {(4,0): 1, (5,1): 1})
    add_candidate({(0,4): 1, (1,4): 1}, {(4,4): 1, (4,5): -1}, {(5,0): 1, (4,1): 1})
    add_candidate({(0,0): 1, (1,4): 1}, {(0,0): 1, (4,5): 1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,0): 1, (1,4): 1}, {(0,0): 1, (4,5): -1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,1): 1, (1,3): 1}, {(1,1): 1, (3,4): 1}, {(1,0): 1, (4,1): 1})
    add_candidate({(0,1): 1, (1,3): 1}, {(1,1): 1, (3,4): -1}, {(4,0): 1, (1,1): 1})
    add_candidate({(0,2): 1, (1,2): 1}, {(2,1): 1, (2,4): 1}, {(1,0): 1, (4,1): 1})
    add_candidate({(0,2): 1, (1,2): 1}, {(2,1): 1, (2,4): -1}, {(4,0): 1, (1,1): 1})
    add_candidate({(0,0): 1, (1,2): 1}, {(0,1): 1, (2,5): 1}, {(1,0): 1, (5,1): 1})
    add_candidate({(0,0): 1, (1,2): 1}, {(0,1): 1, (2,5): -1}, {(5,0): 1, (1,1): 1})
    add_candidate({(0,3): 1, (1,1): 1}, {(3,0): 1, (1,5): 1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,3): 1, (1,1): 1}, {(3,0): 1, (1,5): -1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,4): 1, (1,0): 1}, {(4,2): 1, (0,3): 1}, {(2,0): 1, (3,1): 1})
    add_candidate({(0,4): 1, (1,0): 1}, {(4,2): 1, (0,3): -1}, {(3,0): 1, (2,1): 1})
    add_candidate({(0,2): 1, (1,4): 1}, {(2,0): 1, (4,3): 1}, {(0,0): 1, (3,1): 1})
    add_candidate({(0,2): 1, (1,4): 1}, {(2,0): 1, (4,3): -1}, {(3,0): 1, (0,1): 1})
    add_candidate({(0,1): 1, (1,4): 1}, {(1,2): 1, (4,1): 1}, {(2,0): 1, (1,1): 1})
    add_candidate({(0,1): 1, (1,4): 1}, {(1,2): 1, (4,1): -1}, {(1,0): 1, (2,1): 1})
    add_candidate({(0,3): 1, (1,1): 1}, {(3,3): 1, (1,2): 1}, {(3,0): 1, (2,1): 1})
    add_candidate({(0,3): 1, (1,1): 1}, {(3,3): 1, (1,2): -1}, {(2,0): 1, (3,1): 1})
    add_candidate({(0,0): 1, (1,3): 1}, {(0,4): 1, (3,0): 1}, {(4,0): 1, (0,1): 1})
    add_candidate({(0,0): 1, (1,3): 1}, {(0,4): 1, (3,0): -1}, {(0,0): 1, (4,1): 1})
    add_candidate({(0,4): 1, (1,1): 1}, {(4,3): 1, (1,2): 1}, {(3,0): 1, (2,1): 1})
    add_candidate({(0,4): 1, (1,1): 1}, {(4,3): 1, (1,2): -1}, {(2,0): 1, (3,1): 1})
    add_candidate({(0,2): 1, (1,3): 1}, {(2,4): 1, (3,1): 1}, {(4,0): 1, (1,1): 1})
    add_candidate({(0,2): 1, (1,3): 1}, {(2,4): 1, (3,1): -1}, {(1,0): 1, (4,1): 1})
    add_candidate({(0,1): 1, (1,2): 1}, {(1,3): 1, (2,0): 1}, {(3,0): 1, (0,1): 1})
    add_candidate({(0,1): 1, (1,2): 1}, {(1,3): 1, (2,0): -1}, {(0,0): 1, (3,1): 1})
    add_candidate({(0,3): 1, (1,4): 1}, {(3,5): 1, (4,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,3): 1, (1,4): 1}, {(3,5): 1, (4,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,0): 1, (1,4): 1}, {(0,1): 1, (4,4): 1}, {(1,0): 1, (4,1): 1})
    add_candidate({(0,0): 1, (1,4): 1}, {(0,1): 1, (4,4): -1}, {(4,0): 1, (1,1): 1})
    add_candidate({(0,2): 1, (1,4): 1}, {(2,2): 1, (4,2): 1}, {(2,0): 1, (2,1): 1})
    add_candidate({(0,2): 1, (1,4): 1}, {(2,2): 1, (4,2): -1}, {(2,0): 1, (2,1): -1})
    add_candidate({(0,1): 1, (1,3): 1}, {(1,4): 1, (3,1): 1}, {(4,0): 1, (1,1): 1})
    add_candidate({(0,1): 1, (1,3): 1}, {(1,4): 1, (3,1): -1}, {(1,0): 1, (4,1): 1})
    add_candidate({(0,3): 1, (1,2): 1}, {(3,2): 1, (2,3): 1}, {(2,0): 1, (3,1): 1})
    add_candidate({(0,3): 1, (1,2): 1}, {(3,2): 1, (2,3): -1}, {(3,0): 1, (2,1): 1})
    add_candidate({(0,0): 1, (1,4): 1}, {(0,2): 1, (4,3): 1}, {(2,0): 1, (3,1): 1})
    add_candidate({(0,0): 1, (1,4): 1}, {(0,2): 1, (4,3): -1}, {(3,0): 1, (2,1): 1})
    add_candidate({(0,4): 1, (1,0): 1}, {(4,1): 1, (0,4): 1}, {(1,0): 1, (4,1): 1})
    add_candidate({(0,4): 1, (1,0): 1}, {(4,1): 1, (0,4): -1}, {(4,0): 1, (1,1): 1})
    add_candidate({(0,2): 1, (1,1): 1}, {(2,3): 1, (1,4): 1}, {(3,0): 1, (4,1): 1})
    add_candidate({(0,2): 1, (1,1): 1}, {(2,3): 1, (1,4): -1}, {(4,0): 1, (3,1): 1})
    add_candidate({(0,3): 1, (1,4): 1}, {(3,4): 1, (4,1): 1}, {(4,0): 1, (1,1): 1})
    add_candidate({(0,3): 1, (1,4): 1}, {(3,4): 1, (4,1): -1}, {(1,0): 1, (4,1): 1})
    add_candidate({(0,1): 1, (1,4): 1}, {(1,5): 1, (4,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,1): 1, (1,4): 1}, {(1,5): 1, (4,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,4): 1, (1,1): 1}, {(4,5): 1, (1,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,4): 1, (1,1): 1}, {(4,5): 1, (1,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,2): 1, (1,3): 1}, {(2,5): 1, (3,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,2): 1, (1,3): 1}, {(2,5): 1, (3,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,3): 1, (1,2): 1}, {(3,5): 1, (2,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,3): 1, (1,2): 1}, {(3,5): 1, (2,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,1): 1, (1,2): 1}, {(1,5): 1, (2,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,1): 1, (1,2): 1}, {(1,5): 1, (2,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,4): 1, (1,3): 1}, {(4,5): 1, (3,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,4): 1, (1,3): 1}, {(4,5): 1, (3,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,0): 1, (1,3): 1}, {(0,5): 1, (3,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,0): 1, (1,3): 1}, {(0,5): 1, (3,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,4): 1, (1,2): 1}, {(4,5): 1, (2,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,4): 1, (1,2): 1}, {(4,5): 1, (2,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,3): 1, (1,4): 1}, {(3,5): 1, (4,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,3): 1, (1,4): 1}, {(3,5): 1, (4,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,2): 1, (1,4): 1}, {(2,5): 1, (4,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,2): 1, (1,4): 1}, {(2,5): 1, (4,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,1): 1, (1,4): 1}, {(1,5): 1, (4,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,1): 1, (1,4): 1}, {(1,5): 1, (4,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,0): 1, (1,4): 1}, {(0,5): 1, (4,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,0): 1, (1,4): 1}, {(0,5): 1, (4,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,4): 1, (1,0): 1}, {(4,5): 1, (0,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,4): 1, (1,0): 1}, {(4,5): 1, (0,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,3): 1, (1,1): 1}, {(3,5): 1, (1,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,3): 1, (1,1): 1}, {(3,5): 1, (1,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,2): 1, (1,1): 1}, {(2,5): 1, (1,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,2): 1, (1,1): 1}, {(2,5): 1, (1,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,1): 1, (1,1): 1}, {(1,5): 1, (1,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,1): 1, (1,1): 1}, {(1,5): 1, (1,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,0): 1, (1,1): 1}, {(0,5): 1, (1,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,0): 1, (1,1): 1}, {(0,5): 1, (1,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,4): 1, (1,2): 1}, {(4,5): 1, (2,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,4): 1, (1,2): 1}, {(4,5): 1, (2,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,3): 1, (1,3): 1}, {(3,5): 1, (3,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,3): 1, (1,3): 1}, {(3,5): 1, (3,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,2): 1, (1,2): 1}, {(2,5): 1, (2,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,2): 1, (1,2): 1}, {(2,5): 1, (2,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,1): 1, (1,2): 1}, {(1,5): 1, (2,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,1): 1, (1,2): 1}, {(1,5): 1, (2,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,0): 1, (1,2): 1}, {(0,5): 1, (2,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,0): 1, (1,2): 1}, {(0,5): 1, (2,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,4): 1, (1,3): 1}, {(4,5): 1, (3,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,4): 1, (1,3): 1}, {(4,5): 1, (3,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,3): 1, (1,4): 1}, {(3,5): 1, (4,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,3): 1, (1,4): 1}, {(3,5): 1, (4,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,2): 1, (1,4): 1}, {(2,5): 1, (4,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,2): 1, (1,4): 1}, {(2,5): 1, (4,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,1): 1, (1,4): 1}, {(1,5): 1, (4,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,1): 1, (1,4): 1}, {(1,5): 1, (4,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,0): 1, (1,4): 1}, {(0,5): 1, (4,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,0): 1, (1,4): 1}, {(0,5): 1, (4,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,4): 1, (1,0): 1}, {(4,5): 1, (0,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,4): 1, (1,0): 1}, {(4,5): 1, (0,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,3): 1, (1,1): 1}, {(3,5): 1, (1,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,3): 1, (1,1): 1}, {(3,5): 1, (1,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,2): 1, (1,1): 1}, {(2,5): 1, (1,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,2): 1, (1,1): 1}, {(2,5): 1, (1,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,1): 1, (1,1): 1}, {(1,5): 1, (1,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,1): 1, (1,1): 1}, {(1,5): 1, (1,0): -1}, {(0,0): 1, (5,1): 1})
    add_candidate({(0,0): 1, (1,1): 1}, {(0,5): 1, (1,0): 1}, {(5,0): 1, (0,1): 1})
    add_candidate({(0,0): 1, (1,1): 1}, {(0,5): 1, (1,0): -1}, {(0,0): 1, (5,1): 1})

    num_candidates = len(candidates)
    
    # Precompute contribution matrix
    contrib_matrix = np.zeros((num_candidates, total_entries))
    for idx in range(num_candidates):
        contrib_matrix[idx] = candidates[idx][3]
    
    norms = np.sum(contrib_matrix ** 2, axis=1)
    
    T_target = np.ones(total_entries)
    residual = T_target.copy()
    
    selected_indices = []
    selected_coeffs = []
    
    # Greedy selection
    max_iter = 200
    for _ in range(max_iter):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        dots = contrib_matrix @ residual
        valid = norms > 1e-12
        safe_norms = np.where(valid, norms, 1)
        reductions = np.where(valid, dots**2 / safe_norms, 0)
        best_idx = int(np.argmax(reductions))
        best_reduction = reductions[best_idx]
        
        if best_reduction < 1e-9:
            break
        
        best_coeff = dots[best_idx] / norms[best_idx]
        selected_indices.append(best_idx)
        selected_coeffs.append(best_coeff)
        residual -= best_coeff * contrib_matrix[best_idx]
        
        if np.linalg.norm(residual) < 1e-9:
            break
    
    # Check if we got an exact solution
    if np.linalg.norm(residual) < 1e-9:
        rank_found = len(selected_indices)
        
        U = np.zeros((n * m, rank_found))
        V = np.zeros((m * p, rank_found))
        W = np.zeros((p * n, rank_found))
        
        for r, idx in enumerate(selected_indices):
            coeff = selected_coeffs[r]
            u_vec, v_vec, w_vec, _ = candidates[idx]
            U[:, r] = u_vec * coeff
            V[:, r] = v_vec
            W[:, r] = w_vec
        
        # Try rank reduction via SVD
        def try_reduce_rank(sel_indices, sel_coeffs, U_arr, V_arr, W_arr):
            r = len(sel_indices)
            if r < 2:
                return None
            
            n_entries = n * m * p
            tensor_matrix = np.zeros((n_entries, r))
            for col in range(r):
                idx = sel_indices[col]
                u_vec, v_vec, w_vec, _ = candidates[idx]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            tensor_matrix[i*m*p + j*p + k, col] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
            
            try:
                U_sv, s_vals, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
                tol = 1e-10 * (s_vals[0] if len(s_vals) > 0 else 1)
                null_mask = s_vals < tol
                if not np.any(null_mask):
                    return None
                
                null_space = Vt[null_mask].T
                best_reduced = None
                for ns_vec in null_space:
                    max_idx = int(np.argmax(np.abs(ns_vec)))
                    if abs(ns_vec[max_idx]) < 1e-10:
                        continue
                    
                    new_coeffs = list(sel_coeffs)
                    c_max = new_coeffs[max_idx]
                    for t in range(r):
                        if t != max_idx:
                            new_coeffs[t] = new_coeffs[t] - c_max * ns_vec[t] / ns_vec[max_idx]
                    
                    new_r = r - 1
                    U_new = np.zeros((n * m, new_r))
                    V_new = np.zeros((m * p, new_r))
                    W_new = np.zeros((p * n, new_r))
                    
                    col = 0
                    for t in range(r):
                        if t == max_idx:
                            continue
                        U_new[:, col] = U_arr[:, t]
                        V_new[:, col] = V_arr[:, t]
                        W_new[:, col] = W_arr[:, t]
                        col += 1
                    
                    A_test = np.random.randint(-10, 10, (n, m))
                    B_test = np.random.randint(-10, 10, (m, p))
                    C_test = A_test @ B_test
                    
                    C_calc = np.zeros((n, p))
                    for i in range(n):
                        for k in range(p):
                            val = 0
                            for j in range(m):
                                for rr in range(new_r):
                                    val += U_new[i*m+j, rr] * V_new[j*p+k, rr] * W_new[k*n+i, rr]
                            C_calc[i, k] = val
                    
                    if np.allclose(C_test, C_calc, atol=1e-5):
                        if best_reduced is None or new_r < best_reduced[0]:
                            best_reduced = (new_r, U_new, V_new, W_new)
                
                return best_reduced
            except np.linalg.LinAlgError:
                return None
        
        reduced = try_reduce_rank(selected_indices, selected_coeffs, U, V, W)
        if reduced is not None:
            new_rank, U, V, W = reduced
            rank_found = new_rank
        
        # Verify
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(rank_found):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_calc[i, k] = val
        
        if np.allclose(C_test, C_calc, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return the known 54-rank solution
    n, m, p = 2, 5, 6
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    current_rank = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = -1
        current_rank += 1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = -1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off), current_rank] = -1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = -1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1

    def add_standard_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1

    # Part 1: A[:, 0:2] * B[0:2, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Part 2a: A[:, 2:3] * B[2:3, :]
    for k in range(6):
        add_standard_block(0, 2, k)
        
    # Part 2b: A[:, 3:5] * B[3:5, :]
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
