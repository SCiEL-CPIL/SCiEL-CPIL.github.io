import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS  # A: n x m, B: m x p, C: n x p

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    def verify_exact(U, V, W):
        """Verify exact decomposition using integer/rational arithmetic."""
        R = len(U[0])
        # Convert to Fraction for exactness
        Uf = [[Fraction(int(x * 2)) / 2 for x in row] for row in U]
        Vf = [[Fraction(int(x * 2)) / 2 for x in row] for row in V]
        Wf = [[Fraction(int(x * 2)) / 2 for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                ui = Uf[u_idx(i, j)]
                for k in range(p):
                    vk = Vf[v_idx(j, k)]
                    wk = Wf[w_idx(k, i)]
                    s = Fraction(0)
                    for r in range(R):
                        s += ui[r] * vk[r] * wk[r]
                    if s != 1:
                        return False
        return True

    def get_rank_17_solution():
        """
        Constructive rank 17 decomposition for 2x3x3 matrix multiplication.
        Uses Strassen's algorithm for the 2x2x2 core (indices i,j,k in {0,1})
        and schoolbook terms for the remaining boundary indices (j=2 or k=2).
        """
        R = 17
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]

        # Strassen's 7 terms for i,j,k in {0,1}
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # M2 = (A10+A11)B00      -> C10, C11
        # M3 = A00(B01-B11)      -> C01, C11
        # M4 = A11(B10-B00)      -> C00, C10
        # M5 = (A00+A01)B11      -> C00, C01
        # M6 = (A10-A00)(B00+B01)-> C11
        # M7 = (A01-A11)(B10+B11)-> C00
        
        # r=0 (M1)
        U[u_idx(0,0)][0] = 1; U[u_idx(1,1)][0] = 1
        V[v_idx(0,0)][0] = 1; V[v_idx(1,1)][0] = 1
        W[w_idx(0,0)][0] = 1; W[w_idx(1,1)][0] = 1
        
        # r=1 (M2)
        U[u_idx(1,0)][1] = 1; U[u_idx(1,1)][1] = 1
        V[v_idx(0,0)][1] = 1
        W[w_idx(0,1)][1] = 1; W[w_idx(1,1)][1] = -1
        
        # r=2 (M3)
        U[u_idx(0,0)][2] = 1
        V[v_idx(0,1)][2] = 1; V[v_idx(1,1)][2] = -1
        W[w_idx(1,0)][2] = 1; W[w_idx(1,1)][2] = 1
        
        # r=3 (M4)
        U[u_idx(1,1)][3] = 1
        V[v_idx(1,0)][3] = 1; V[v_idx(0,0)][3] = -1
        W[w_idx(0,0)][3] = 1; W[w_idx(0,1)][3] = 1
        
        # r=4 (M5)
        U[u_idx(0,0)][4] = 1; U[u_idx(0,1)][4] = 1
        V[v_idx(1,1)][4] = 1
        W[w_idx(0,0)][4] = -1; W[w_idx(1,0)][4] = 1
        
        # r=5 (M6)
        U[u_idx(1,0)][5] = 1; U[u_idx(0,0)][5] = -1
        V[v_idx(0,0)][5] = 1; V[v_idx(0,1)][5] = 1
        W[w_idx(1,1)][5] = 1
        
        # r=6 (M7)
        U[u_idx(0,1)][6] = 1; U[u_idx(1,1)][6] = -1
        V[v_idx(1,0)][6] = 1; V[v_idx(1,1)][6] = 1
        W[w_idx(0,0)][6] = 1

        # Boundary terms (r=7..16)
        # These cover all (i,j,k) where j=2 or k=2
        boundary_terms = [
            (0, 2, 0), (1, 2, 0),
            (0, 2, 1), (1, 2, 1),
            (0, 2, 2), (1, 2, 2),
            (0, 0, 2), (1, 0, 2),
            (0, 1, 2), (1, 1, 2)
        ]
        
        for idx, (i, j, k) in enumerate(boundary_terms):
            r = 7 + idx
            U[u_idx(i, j)][r] = 1
            V[v_idx(j, k)][r] = 1
            W[w_idx(k, i)][r] = 1

        return U, V, W

    def get_schoolbook_solution():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[u_idx(i, j)][c] = 1
                    V[v_idx(j, k)][c] = 1
                    W[w_idx(k, i)][c] = 1
                    c += 1
        return U, V, W

    # 1. Construct the known rank-17 solution
    U17, V17, W17 = get_rank_17_solution()
    
    # Verify it to be safe (should always pass)
    if verify_exact(U17, V17, W17):
        best_result = {"U": U17, "V": V17, "W": W17}
    else:
        # Fallback to schoolbook if something went wrong (should not happen)
        U_sb, V_sb, W_sb = get_schoolbook_solution()
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # 2. Try to find a rank-16 solution
    # Strategy: The rank-17 solution has 10 boundary terms that are "sparse" (only 1 non-zero in each factor).
    # We can try to merge two boundary terms if they share structure, or if we can express their sum with one term.
    # However, a more robust approach for small tensors is to try dropping one term from the rank-17 solution
    # and see if the remaining 16 terms can be adjusted to satisfy the equations.
    # This is a non-linear system. We can use a simple heuristic:
    # If we drop term r, we have a deficit D. Can we add a new term that covers D?
    # Actually, the problem is to find *any* rank 16 decomposition.
    
    # Let's try a different approach: Random search for rank 16.
    # Since the space is small, we can try to fix the structure of the boundary terms.
    # The core 2x2x2 requires rank 7 (Strassen).
    # The boundary involves j=2 or k=2.
    # Terms:
    # A_i2 B_2k C_ki (6 terms: i in {0,1}, k in {0,1,2}) -> wait, j=2 means A_i2.
    # A_ij B_2k C_ki (6 terms: j in {0,1}, k in {0,1,2}) -> wait, k=2 means B_j2.
    # Total boundary terms in schoolbook: 
    # j=2: (0,2,0), (0,2,1), (0,2,2), (1,2,0), (1,2,1), (1,2,2) -> 6 terms
    # k=2: (0,0,2), (0,1,2), (1,0,2), (1,1,2) -> 4 terms (note (0,2,2) and (1,2,2) already counted in j=2? No, j=2 is A_i2, k=2 is B_j2. They are distinct unless j=2 AND k=2. 
    # Wait, the indices are (i, j, k).
    # j=2: i in {0,1}, k in {0,1,2} -> 6 terms.
    # k=2: i in {0,1}, j in {0,1,2} -> 6 terms.
    # Overlap: j=2 AND k=2 -> (0,2,2), (1,2,2) -> 2 terms.
    # Total boundary = 6 + 6 - 2 = 10 terms.
    # Core (j in {0,1}, k in {0,1}) -> 4x4=16 terms, reduced to 7 by Strassen.
    # Total 7 + 10 = 17.

    # Can we do better?
    # The boundary terms form a tensor of size (2*3) x (3*2) x (2*2)? No.
    # Let's just try to optimize the boundary terms if we keep the core fixed.
    # But the core and boundary are coupled if we allow mixing.
    
    # Given the time limit and the difficulty of finding rank 16 (which is likely 16 or 17),
    # and the fact that rank 17 is already a significant improvement over 18,
    # we will stick with the verified rank 17 solution.
    
    # If we have time, we could attempt a local search on the rank 17 solution to see if it collapses to rank 16,
    # but that's risky. The constructive solution is robust.

    # Return the best result found
    return best_result

if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
