import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W):
    """Verify exact tensor decomposition using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for M in (U, V, W):
        for row in M:
            if len(row) != rank:
                return False
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    one = Fraction(1, 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                urow = Uf[i * m + j]
                vrow = Vf[j * p + k]
                wrow = Wf[k * n + i]
                for r in range(rank):
                    s += urow[r] * vrow[r] * wrow[r]
                if s != one:
                    return False
    return True

def _build_rank88():
    """Build the known rank-88 decomposition."""
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part 1: i=0, schoolbook 1x4x8
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2 via Strassen 2x2x2 blocks
    sU = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
    ]

    for jb in range(2):
        j0 = jb * 2
        for kb in range(4):
            k0 = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[(1 + li) * m + (j0 + lj)][col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2
                        V[(j0 + lj) * p + (k0 + lk)][col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2
                        W[(k0 + lk) * n + (1 + li)][col] += sW[t][l]
                col += 1

    return {"U": U, "V": V, "W": W}

def _build_laderman():
    """
    Build Laderman rank-14 decomposition for 2x3x3 matrix multiplication.
    Convention: U[i*3+j], V[j*3+k], W[k*2+i]
    """
    n, m, p = 2, 3, 3
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Laderman's 14 terms
    # Term format: (u_indices, v_indices, w_indices) where each is list of (index, coeff)
    # We'll define the 14 rank-1 terms explicitly

    # Let's use the standard Laderman decomposition
    # M = [[a,b,c],[d,e,f]] * N = [[g,h,i],[j,k,l]] = [[p,q,r],[s,t,u]]
    
    # The 14 products are:
    # 1: (a+e-f-j)*(g+k-l-i) -> p term? Let me use the standard formulation.
    
    # Actually, let me construct it from the known rank-14 decomposition.
    # A simpler approach: use the known Laderman matrices.
    
    # Laderman decomposition for 2x3x3:
    # The 14 tensors are defined as follows (using 0-indexed):
    # A indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    # C indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    
    # Let me just hardcode a valid Laderman decomposition.
    # From literature, one valid set of 14 terms:
    
    terms = [
        # (U_support, V_support, W_support)
        # Each support is dict: {index: coeff}
        ({0:1, 4:1, 5:-1}, {0:1, 4:1, 5:-1}, {0:1}),
        ({0:1, 4:1, 5:-1}, {0:1, 3:1, 5:-1}, {1:1}),
        ({0:1, 4:1, 5:-1}, {0:1, 3:1, 5:-1}, {2:1}),
        ({1:1, 4:1, 5:-1}, {1:1, 4:1, 5:-1}, {3:1}),
        ({1:1, 4:1, 5:-1}, {1:1, 3:1, 5:-1}, {4:1}),
        ({1:1, 4:1, 5:-1}, {1:1, 3:1, 5:-1}, {5:1}),
        ({2:1, 4:1, 5:-1}, {2:1, 4:1, 5:-1}, {0:1}),
        ({2:1, 4:1, 5:-1}, {2:1, 3:1, 5:-1}, {1:1}),
        ({2:1, 4:1, 5:-1}, {2:1, 3:1, 5:-1}, {2:1}),
        ({3:1, 4:1, 5:-1}, {3:1, 4:1, 5:-1}, {3:1}),
        ({3:1, 4:1, 5:-1}, {3:1, 3:1, 5:-1}, {4:1}),
        ({3:1, 4:1, 5:-1}, {3:1, 3:1, 5:-1}, {5:1}),
        ({4:1}, {4:1}, {4:1}),
        ({5:-1}, {5:-1}, {5:1}),
    ]
    
    # This is getting complicated. Let me use a different approach.
    # I'll construct Laderman's decomposition directly from its definition.
    
    # Actually, let me just use a known valid Laderman decomposition.
    # Here's one from the literature (rank 14 for 2x3x3):
    
    U = [[0]*14 for _ in range(6)]
    V = [[0]*14 for _ in range(9)]
    W = [[0]*14 for _ in range(6)]
    
    # Using a standard Laderman decomposition:
    # I'll define it using the standard formulation.
    
    # For now, let me just return the Strassen-based rank-88 solution
    # and focus on the overall structure.
    
    # Let me try a different approach: construct Laderman properly.
    
    # The Laderman decomposition for 2x3x3:
    # Let A = [[a00, a01, a02], [a10, a11, a12]]
    # Let B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C = A * B, a 2x3 matrix
    
    # The 14 bilinear forms are:
    # I'll use a known valid set.
    
    # For simplicity, let me just build the rank-88 solution and return it.
    # The key improvement would come from actually implementing Laderman correctly.
    
    # Let me implement Laderman step by step.
    
    # Actually, I realize I should just focus on producing a valid solution.
    # Let me return the rank-88 solution for now and add the Laderman improvement
    # in a cleaner way.
    
    raise NotImplementedError("Laderman decomposition not yet implemented")

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    # Start with the known rank-88 solution
    best = _build_rank88()
    
    if not _verify(best["U"], best["V"], best["W"]):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        best = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return best
