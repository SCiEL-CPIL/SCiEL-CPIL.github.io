"""Recursive Strassen-based decomposition for 2x3x8 matrix multiplication."""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
    def _schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2 decomposition
    def _strassen_222():
        S_U = [
            [1, 0, 0, 1], 
            [0, 0, 1, 1], 
            [1, 0, 0, 0], 
            [0, 0, 0, 1], 
            [1, 1, 0, 0], 
            [-1, 0, 1, 0],
            [0, 1, 0, -1] 
        ]
        S_V = [
            [1, 0, 0, 1], 
            [1, 0, 0, 0], 
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1], 
            [1, 1, 0, 0], 
            [0, 0, 1, 1]  
        ]
        S_W = [
            [1, 0, 0, 1], 
            [0, 1, 0, -1],
            [0, 0, 1, 1], 
            [1, 1, 0, 0], 
            [-1, 0, 1, 0],
            [0, 0, 0, 1], 
            [1, 0, 0, 0]  
        ]
        return {"U": S_U, "V": S_V, "W": S_W}

    def _combine_decompositions(decs, n, m, p, offsets):
        """
        Combine a list of decompositions for sub-problems.
        offsets: list of (i_off, a_off, k_off) for each dec.
        """
        total_rank = sum(len(d["U"][0]) for d in decs)
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col_offset = 0
        for idx, dec in enumerate(decs):
            i_off, a_off, k_off = offsets[idx]
            rank_sub = len(dec["U"][0])
            # Map sub-indices to global indices
            # Sub U is (n_sub x m_sub), global row = i_off * m_sub + i_sub? 
            # No, the dimensions of the sub-problem are defined by the split.
            # This helper is too complex for variable splits. 
            # We will implement specific recursive splits for 2x3x8.
            pass
        return {"U": U, "V": V, "W": W}

    # Specific recursive construction for 2x3x8
    # Split 8 into 4+4
    # Each 2x3x4 split into 2x2x4 + 2x1x4
    # 2x2x4 split into two 2x2x2
    # 2x1x4 is schoolbook rank 8
    
    def _build_2x2x2(n, m, p):
        return _strassen_222()

    def _build_2x1x4(n, m, p):
        # n=2, m=1, p=4
        rank = n * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i * m][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return {"U": U, "V": V, "W": W}

    def _build_2x2x4(n, m, p):
        # Split p into 2+2
        # Dec 1: 2x2x2 (k=0,1)
        # Dec 2: 2x2x2 (k=2,3)
        d1 = _build_2x2x2(n, m, 2)
        d2 = _build_2x2x2(n, m, 2)
        
        rank1 = len(d1["U"][0])
        rank2 = len(d2["U"][0])
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Copy d1
        for i in range(n * m):
            for c in range(rank1):
                U[i][c] = d1["U"][i][c]
        for a in range(m):
            for k in range(2): # p_sub = 2
                for c in range(rank1):
                    V[a * p + k][c] = d1["V"][a * 2 + k][c]
        for k in range(2):
            for i in range(n):
                for c in range(rank1):
                    W[k * n + i][c] = d1["W"][k * n + i][c]
        
        # Copy d2 (offset rank1, k offset 2)
        for i in range(n * m):
            for c in range(rank2):
                U[i][rank1 + c] = d2["U"][i][c]
        for a in range(m):
            for k in range(2):
                for c in range(rank2):
                    V[a * p + (2 + k)][rank1 + c] = d2["V"][a * 2 + k][c]
        for k in range(2):
            for i in range(n):
                for c in range(rank2):
                    W[(2 + k) * n + i][rank1 + c] = d2["W"][k * n + i][c]
                    
        return {"U": U, "V": V, "W": W}

    def _build_2x3x4(n, m, p):
        # Split m into 2+1
        # Dec 1: 2x2x4 (a=0,1)
        # Dec 2: 2x1x4 (a=2)
        d1 = _build_2x2x4(n, 2, p)
        d2 = _build_2x1x4(n, 1, p)
        
        rank1 = len(d1["U"][0])
        rank2 = len(d2["U"][0])
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Copy d1 (a offset 0)
        for i in range(n):
            for a in range(2):
                for c in range(rank1):
                    U[i * m + a][c] = d1["U"][i * 2 + a][c]
        for a in range(2):
            for k in range(p):
                for c in range(rank1):
                    V[a * p + k][c] = d1["V"][a * p + k][c]
        for k in range(p):
            for i in range(n):
                for c in range(rank1):
                    W[k * n + i][c] = d1["W"][k * n + i][c]
        
        # Copy d2 (a offset 2)
        for i in range(n):
            for a in range(1):
                for c in range(rank2):
                    U[i * m + (2 + a)][rank1 + c] = d2["U"][i * 1 + a][c]
        for a in range(1):
            for k in range(p):
                for c in range(rank2):
                    V[(2 + a) * p + k][rank1 + c] = d2["V"][a * p + k][c]
        for k in range(p):
            for i in range(n):
                for c in range(rank2):
                    W[k * n + i][rank1 + c] = d2["W"][k * n + i][c]
                    
        return {"U": U, "V": V, "W": W}

    def _build_2x3x8(n, m, p):
        # Split p into 4+4
        d1 = _build_2x3x4(n, m, 4)
        d2 = _build_2x3x4(n, m, 4)
        
        rank1 = len(d1["U"][0])
        rank2 = len(d2["U"][0])
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Copy d1 (k offset 0)
        for i in range(n * m):
            for c in range(rank1):
                U[i][c] = d1["U"][i][c]
        for a in range(m):
            for k in range(4):
                for c in range(rank1):
                    V[a * p + k][c] = d1["V"][a * 4 + k][c]
        for k in range(4):
            for i in range(n):
                for c in range(rank1):
                    W[k * n + i][c] = d1["W"][k * n + i][c]
        
        # Copy d2 (k offset 4)
        for i in range(n * m):
            for c in range(rank2):
                U[i][rank1 + c] = d2["U"][i][c]
        for a in range(m):
            for k in range(4):
                for c in range(rank2):
                    V[a * p + (4 + k)][rank1 + c] = d2["V"][a * 4 + k][c]
        for k in range(4):
            for i in range(n):
                for c in range(rank2):
                    W[(4 + k) * n + i][rank1 + c] = d2["W"][k * n + i][c]
                    
        return {"U": U, "V": V, "W": W}

    candidate = _build_2x3x8(n, m, p)
    
    # Validation
    def _validate(candidate):
        from fractions import Fraction
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
