import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 115  # Safety margin before 120s limit

    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # Rank-50 construction using Strassen for 2x2 blocks
    for block_start in range(0, 8, 2):
        k0 = block_start
        k1 = block_start + 1
        
        add_term(
            {0: 1, 4: 1},
            {k0: 1, p+k1: 1},
            {k0*n: 1, k1*n+1: 1}
        )
        add_term(
            {3: 1, 4: 1},
            {k0: 1},
            {k0*n+1: 1, k1*n+1: -1}
        )
        add_term(
            {0: 1},
            {k1: 1, p+k1: -1},
            {k1*n: 1, k1*n+1: 1}
        )
        add_term(
            {4: 1},
            {p+k0: 1, k0: -1},
            {k0*n: 1, k0*n+1: 1}
        )
        add_term(
            {0: 1, 1: 1},
            {p+k1: 1},
            {k0*n: -1, k1*n: 1}
        )
        add_term(
            {3: 1, 0: -1},
            {k0: 1, k1: 1},
            {k1*n+1: 1}
        )
        add_term(
            {1: 1, 4: -1},
            {p+k0: 1, p+k1: 1},
            {k0*n: 1}
        )
    
    k = 8
    add_term({0: 1}, {k: 1}, {k*n: 1})
    add_term({1: 1}, {p+k: 1}, {k*n: 1})
    add_term({3: 1}, {k: 1}, {k*n+1: 1})
    add_term({4: 1}, {p+k: 1}, {k*n+1: 1})
    
    for i in range(n):
        for k in range(p):
            add_term(
                {i*m+2: 1},
                {2*p+k: 1},
                {k*n+i: 1}
            )
            
    rank = len(U_cols)
    
    def verify_decomposition(U_cols, V_cols, W_cols):
        """Verify that the decomposition exactly reproduces the multiplication tensor."""
        r = len(U_cols)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    val = 0
                    for s in range(r):
                        val += U_cols[s][a] * V_cols[s][b] * W_cols[s][c]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    def try_remove_terms(indices_to_remove):
        """Try to remove given terms by finding coefficients for remaining terms."""
        if time.time() > deadline - 3:
            return None
        
        r = len(U_cols)
        remaining = [s for s in range(r) if s not in indices_to_remove]
        num_remaining = len(remaining)
        
        if num_remaining == 0:
            return None
        
        # Build the linear system: find d_s for s in remaining
        # such that sum_s d_s * U_s * V_s * W_s = sum_s U_s * V_s * W_s (full identity)
        # This means: sum_s (d_s - 1) * (U_s * V_s * W_s) = -sum_{s in removed} (U_s * V_s * W_s)
        # But we want the FULL sum to equal identity, so:
        # sum_{s in remaining} d_s * (U_s[a] * V_s[b] * W_s[c]) = 1 for all (a,b,c)
        
        # Number of equations: n*m*p = 2*3*9 = 54
        # Number of unknowns: num_remaining
        
        equations_A = []
        equations_b = []
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    row = np.zeros(num_remaining)
                    for idx, s in enumerate(remaining):
                        row[idx] = U_cols[s][a] * V_cols[s][b] * W_cols[s][c]
                    equations_A.append(row)
                    equations_b.append(1.0)
        
        A_sys = np.array(equations_A)
        b_sys = np.array(equations_b)
        
        # Solve the underdetermined system (54 equations, num_remaining unknowns)
        # Use least squares to find a solution, then check if it's exact
        try:
            d, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            
            # Check if the solution is exact
            A_sys_d = A_sys @ d
            if not np.allclose(A_sys_d, b_sys, atol=1e-10):
                return None
            
            # Construct new decomposition
            new_U_cols = [U_cols[s][:] for s in remaining]
            new_V_cols = [V_cols[s][:] for s in remaining]
            new_W_cols = [W_cols[s][:] for s in remaining]
            
            # Scale each term by its coefficient d
            for idx, s in enumerate(remaining):
                scale = d[idx]
                if abs(scale) < 1e-15:
                    continue
                for i in range(n*m):
                    new_U_cols[idx][i] *= scale
                for i in range(m*p):
                    new_V_cols[idx][i] *= scale
                for i in range(p*n):
                    new_W_cols[idx][i] *= scale
            
            # Final verification
            if verify_decomposition(new_U_cols, new_V_cols, new_W_cols):
                return (new_U_cols, new_V_cols, new_W_cols)
        except:
            pass
        
        return None
    
    # Try removing 1 term
    for r_remove in range(rank):
        if time.time() > deadline - 10:
            break
        res = try_remove_terms([r_remove])
        if res:
            U_cols, V_cols, W_cols = res
            rank = len(U_cols)
            break
    
    # Try removing 2 terms
    if rank > 40 and time.time() < deadline - 15:
        for r1, r2 in itertools.combinations(range(rank), 2):
            if time.time() > deadline - 5:
                break
            res = try_remove_terms([r1, r2])
            if res:
                U_cols, V_cols, W_cols = res
                rank = len(U_cols)
                break
    
    # Try removing 3 terms (only if we have time)
    if rank > 40 and time.time() < deadline - 20:
        # Try a limited number of triples
        triples_tried = 0
        for r1, r2, r3 in itertools.combinations(range(rank), 3):
            if time.time() > deadline - 5 or triples_tried >= 50:
                break
            triples_tried += 1
            res = try_remove_terms([r1, r2, r3])
            if res:
                U_cols, V_cols, W_cols = res
                rank = len(U_cols)
                break
    
    # Final cleanup
    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
