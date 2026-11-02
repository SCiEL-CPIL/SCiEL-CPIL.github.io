import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 105
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Strassen 7-term decomposition for 2x2x2
    # Verified: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k in {0,1}
    STRASSEN = [
        # (U entries {(i,j):val}, V entries {(j,k):val}, W entries {(k,i):val})
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(0,0):1}, {(0,0):-1, (1,0):1}, {(0,0):-1, (1,0):1}),
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(1,0):1, (1,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1, (1,0):-1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    def build_decomposition(j_parts, k_parts):
        """Build decomposition given partitions of j and k dimensions.
        j_parts: list of (start, end) for j dimension
        k_parts: list of (start, end) for k dimension
        Returns (terms, rank) or None if invalid.
        """
        terms = []
        
        for (j0, j1) in j_parts:
            for (k0, k1) in k_parts:
                jm = j1 - j0
                km = k1 - k0
                if jm == 2 and km == 2:
                    # Use Strassen: 7 terms
                    for (u_d, v_d, w_d) in STRASSEN:
                        u_abs = {}
                        for (di, dj), val in u_d.items():
                            u_abs[(di, j0+dj)] = val
                        v_abs = {}
                        for (dj, dk), val in v_d.items():
                            v_abs[(j0+dj, k0+dk)] = val
                        w_abs = {}
                        for (dk, di), val in w_d.items():
                            w_abs[(k0+dk, di)] = val
                        terms.append((u_abs, v_abs, w_abs))
                elif jm == 1 and km == 1:
                    # Schoolbook: 2*1*1 = 2 terms
                    for di in range(2):
                        u = {(di, j0): 1}
                        v = {(j0, k0): 1}
                        w = {(k0, di): 1}
                        terms.append((u, v, w))
                elif jm == 2 and km == 1:
                    # Schoolbook: 2*2*1 = 4 terms
                    for di in range(2):
                        for dj in range(2):
                            u = {(di, j0+dj): 1}
                            v = {(j0+dj, k0): 1}
                            w = {(k0, di): 1}
                            terms.append((u, v, w))
                elif jm == 1 and km == 2:
                    # Schoolbook: 2*1*2 = 4 terms
                    for di in range(2):
                        for dk in range(2):
                            u = {(di, j0): 1}
                            v = {(j0, k0+dk): 1}
                            w = {(k0+dk, di): 1}
                            terms.append((u, v, w))
        
        rank = len(terms)
        
        # Build and verify
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for r in range(rank):
            u_dict, v_dict, w_dict = terms[r]
            for (i, j), val in u_dict.items():
                U[i*m + j, r] = val
            for (j, k), val in v_dict.items():
                V[j*p + k, r] = val
            for (k, i), val in w_dict.items():
                W[k*n + i, r] = val
        
        # Verify
        max_error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    error = abs(val - 1)
                    if error > max_error:
                        max_error = error
        
        if max_error > 1e-9:
            return None
        
        return (U, V, W, rank)
    
    # Try different partitions
    # All possible ways to partition {0..4} into segments of size 1 or 2
    def gen_partitions(size):
        """Generate all partitions of range(size) into contiguous segments of size 1 or 2."""
        if size == 0:
            yield []
            return
        # First segment is size 1
        for rest in gen_partitions(size - 1):
            yield [(0, 1)] + [(a+1, b+1) for (a, b) in rest]
        # First segment is size 2
        if size >= 2:
            for rest in gen_partitions(size - 2):
                yield [(0, 2)] + [(a+2, b+2) for (a, b) in rest]
    
    best_result = None
    best_rank = 999999
    
    j_options = list(gen_partitions(5))
    k_options = list(gen_partitions(5))
    
    for j_parts in j_options:
        for k_parts in k_options:
            result = build_decomposition(j_parts, k_parts)
            if result is not None:
                U, V, W, rank = result
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U, V, W, rank)
                    if time.time() > deadline - 10:
                        break
        if time.time() > deadline - 10:
            break
    
    if best_result is None:
        # Fallback: pure schoolbook
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))
        rank = len(terms)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            u_dict, v_dict, w_dict = terms[r]
            for (i, j), val in u_dict.items():
                U[i*m + j, r] = val
            for (j, k), val in v_dict.items():
                V[j*p + k, r] = val
            for (k, i), val in w_dict.items():
                W[k*n + i, r] = val
        best_result = (U, V, W, rank)
    
    U, V, W, rank = best_result
    
    # Convert to list format
    def clean_matrix(mat):
        result = []
        for row in mat:
            cleaned_row = []
            for val in row:
                fv = float(val)
                if abs(fv - round(fv)) < 1e-9:
                    cleaned_row.append(int(round(fv)))
                else:
                    cleaned_row.append(fv)
            result.append(cleaned_row)
        return result
    
    return {"U": clean_matrix(U), "V": clean_matrix(V), "W": clean_matrix(W)}
