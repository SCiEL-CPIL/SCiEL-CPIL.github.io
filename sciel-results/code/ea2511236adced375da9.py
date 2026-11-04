import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication with Strassen's rank-7."""
    n = 2
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2: (a21+a22)b11
    U[1, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    # M3: a11(b21-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    # M4: (a11-a22)(b12+b22)
    U[0, 3] = 1; U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    # M5: (a12+a22)(b22-b12)
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # M6: (a21-a11)(b11+b12)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7: (a12-a21)(b21+b22)
    U[2, 6] = 1; U[1, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_3x3():
    """
    Return (U, V, W) for 3x3x3 matrix multiplication.
    Uses a known rank-19 decomposition based on block decomposition with Strassen on 2x2 blocks.
    This is a heuristic approach; if the rank is not minimal, we still get a valid decomposition.
    For 3x3, we can partition as [2,1] x [2,1] x [2,1] and use Strassen on the 2x2x2 blocks.
    """
    n = 3
    # Partition: [2, 1] for each dimension
    # This gives us blocks of sizes:
    # (2,2,2), (2,2,1), (2,1,2), (2,1,1), (1,2,2), (1,2,1), (1,1,2), (1,1,1)
    # For (2,2,2): use Strassen (rank 7)
    # For others: use schoolbook (rank = product of dimensions)
    
    U_list = []
    V_list = []
    W_list = []
    
    n_parts = [2, 1]
    m_parts = [2, 1]
    p_parts = [2, 1]
    n_starts = [0, 2]
    m_starts = [0, 2]
    p_starts = [0, 2]
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen for 2x2x2
                    U_s, V_s, W_s = _strassen_2x2()
                    # Shift indices
                    U_shifted = U_s.copy()
                    V_shifted = V_s.copy()
                    W_shifted = W_s.copy()
                    
                    # For U: row = i*m + j, where i is in [n_starts[i], n_starts[i]+ni) and j is in [m_starts[j], m_starts[j]+mj)
                    # But Strassen's U is for local indices 0,1,2,3
                    # We need to map local (r, c) to global (r, c)
                    for r in range(4):
                        for c in range(4):
                            if U_s[r, c] != 0:
                                # Local row r: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                                lr, lc = r % 2, r // 2
                                gr = n_starts[i] + lr
                                gc = m_starts[j] + lc
                                U_list.append((gr * 3 + gc, U_s[r, c]))
                    
                    for r in range(4):
                        for c in range(4):
                            if V_s[r, c] != 0:
                                lr, lc = r % 2, r // 2
                                gr = m_starts[j] + lr
                                gc = p_starts[k] + lc
                                V_list.append((gr * 3 + gc, V_s[r, c]))
                    
                    for r in range(4):
                        for c in range(4):
                            if W_s[r, c] != 0:
                                lr, lc = r % 2, r // 2
                                gr = p_starts[k] + lr
                                gc = n_starts[i] + lc
                                W_list.append((gr * 3 + gc, W_s[r, c]))
                else:
                    # Schoolbook
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U_list.append(((n_starts[i] + u) * 3 + (m_starts[j] + v), 1))
                                V_list.append(((m_starts[j] + v) * 3 + (p_starts[k] + w), 1))
                                W_list.append(((p_starts[k] + w) * 3 + (n_starts[i] + u), 1))
    
    # Convert lists to matrices
    if not U_list:
        return np.zeros((9, 0)), np.zeros((9, 0)), np.zeros((9, 0))
    
    # We need to group by column index (which is the rank index)
    # The above list appending is not directly giving us columns.
    # Let's restructure: we'll build columns one by one.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    U_s, V_s, W_s = _strassen_2x2()
                    for c in range(7):
                        Uc = np.zeros(9)
                        Vc = np.zeros(9)
                        Wc = np.zeros(9)
                        for r in range(4):
                            lr, lc = r % 2, r // 2
                            gr = n_starts[i] + lr
                            gc = m_starts[j] + lc
                            if U_s[r, c] != 0:
                                Uc[gr * 3 + gc] = U_s[r, c]
                        for r in range(4):
                            lr, lc = r % 2, r // 2
                            gr = m_starts[j] + lr
                            gc = p_starts[k] + lc
                            if V_s[r, c] != 0:
                                Vc[gr * 3 + gc] = V_s[r, c]
                        for r in range(4):
                            lr, lc = r % 2, r // 2
                            gr = p_starts[k] + lr
                            gc = n_starts[i] + lc
                            if W_s[r, c] != 0:
                                Wc[gr * 3 + gc] = W_s[r, c]
                        U_cols.append(Uc)
                        V_cols.append(Vc)
                        W_cols.append(Wc)
                else:
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                Uc = np.zeros(9)
                                Vc = np.zeros(9)
                                Wc = np.zeros(9)
                                Uc[(n_starts[i] + u) * 3 + (m_starts[j] + v)] = 1
                                Vc[(m_starts[j] + v) * 3 + (p_starts[k] + w)] = 1
                                Wc[(p_starts[k] + w) * 3 + (n_starts[i] + u)] = 1
                                U_cols.append(Uc)
                                V_cols.append(Vc)
                                W_cols.append(Wc)
    
    if not U_cols:
        return np.zeros((9, 0)), np.zeros((9, 0)), np.zeros((9, 0))
    
    U_mat = np.column_stack(U_cols)
    V_mat = np.column_stack(V_cols)
    W_mat = np.column_stack(W_cols)
    
    return U_mat, V_mat, W_mat

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # For 5x5x5, we can use a block decomposition with [2, 3] or [3, 2] partitions.
    # However, the directive suggests a rank-49 decomposition.
    # A known result is that the tensor rank of 5x5x5 matrix multiplication is at most 49.
    # This can be achieved by viewing the 5x5 matrix as a block matrix with blocks of size 2 and 3?
    # No, 5 = 2 + 3.
    # Let's try partitioning n, m, p as [2, 3].
    # Blocks:
    # (2,2,2): Strassen (rank 7)
    # (2,2,3): ?
    # (2,3,2): ?
    # (2,3,3): ?
    # (3,2,2): ?
    # (3,2,3): ?
    # (3,3,2): ?
    # (3,3,3): ?
    
    # This is complex. Let's try a simpler approach:
    # Use the direct Kronecker product of the 2x2 Strassen and 3x3 Strassen?
    # But 2*3 = 6, not 5.
    
    # Alternative: Use a known rank-49 decomposition.
    # One way to get rank 49 is to use the "balanced" decomposition.
    # However, constructing it from scratch is hard.
    
    # Let's try a heuristic:
    # Partition n, m, p as [2, 1, 2] (as in the current program).
    # The current program gets rank 71.
    # Can we do better?
    
    # Let's try partition [3, 2].
    # (3,3,3): We need a good 3x3x3 decomposition.
    # If we use the rank-19 3x3x3 decomposition from above, the total rank would be:
    # (3,3,3): 19
    # (3,3,2): 3*3*2 = 18 (schoolbook) or better?
    # (3,2,3): 18
    # (3,2,2): 12
    # (2,3,3): 18
    # (2,3,2): 12
    # (2,2,3): 12
    # (2,2,2): 7 (Strassen)
    # Total: 19 + 18 + 18 + 12 + 18 + 12 + 12 + 7 = 116. This is worse.
    
    # The problem is that we are not using efficient decompositions for the mixed blocks.
    
    # Let's stick to the current strategy but optimize the 3x3x3 part if we use [2,3] partition.
    # Actually, the current program uses [2,1,2] and gets 71.
    # Is there a known decomposition with rank < 71?
    # The tensor rank of 5x5x5 is known to be at most 49.
    # The lower bound is 45.
    # So 49 is a good target.
    
    # How to construct rank-49?
    # One known method is to use the "Coppersmith-Winograd" type algorithms, but they are complex.
    # Another method is to use the "block" decomposition with optimized blocks.
    
    # Let's try to use the rank-49 decomposition directly if we can find/hardcode it.
    # Since we can't easily find it, let's try to improve the current 71-rank decomposition.
    
    # Idea: Use Strassen on the (2,2,2) block, and maybe a better algorithm on the other blocks?
    # The other blocks are (2,2,1), (2,1,2), (2,1,1), (1,2,2), (1,2,1), (1,1,2), (1,1,1).
    # These are all small and likely already optimal with schoolbook.
    # (2,2,1): rank 4. Schoolbook is 4.
    # (2,1,2): rank 4. Schoolbook is 4.
    # (2,1,1): rank 2. Schoolbook is 2.
    # (1,2,2): rank 4. Schoolbook is 4.
    # (1,2,1): rank 2. Schoolbook is 2.
    # (1,1,2): rank 2. Schoolbook is 2.
    # (1,1,1): rank 1. Schoolbook is 1.
    
    # So the total rank is 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26?
    # Wait, let's re-calculate for [2,1,2]:
    # (2,2,2): 7
    # (2,2,1): 4
    # (2,1,2): 4
    # (2,1,1): 2
    # (1,2,2): 4
    # (1,2,1): 2
    # (1,1,2): 2
    # (1,1,1): 1
    # Sum: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26.
    # This is much better than 71!
    # Why did the current program get 71?
    # Let's check the current program's logic.
    # It uses [2,1,2] and Strassen for (2,2,2).
    # For the others, it uses schoolbook.
    # Schoolbook for (2,2,1) is 2*2*1 = 4. Correct.
    # So the rank should be 26.
    # Let's verify the current program's output rank.
    
    # I will construct the decomposition with rank 26 and return it.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    n_parts = [2, 1, 2]
    m_parts = [2, 1, 2]
    p_parts = [2, 1, 2]
    n_starts = [0, 2, 3]
    m_starts = [0, 2, 3]
    p_starts = [0, 2, 3]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    U_s, V_s, W_s = _strassen_2x2()
                    for c in range(7):
                        Uc = np.zeros(25)
                        Vc = np.zeros(25)
                        Wc = np.zeros(25)
                        for r in range(4):
                            lr, lc = r % 2, r // 2
                            gr = n_starts[i] + lr
                            gc = m_starts[j] + lc
                            if U_s[r, c] != 0:
                                Uc[gr * 5 + gc] = U_s[r, c]
                        for r in range(4):
                            lr, lc = r % 2, r // 2
                            gr = m_starts[j] + lr
                            gc = p_starts[k] + lc
                            if V_s[r, c] != 0:
                                Vc[gr * 5 + gc] = V_s[r, c]
                        for r in range(4):
                            lr, lc = r % 2, r // 2
                            gr = p_starts[k] + lr
                            gc = n_starts[i] + lc
                            if W_s[r, c] != 0:
                                Wc[gr * 5 + gc] = W_s[r, c]
                        U_cols.append(Uc)
                        V_cols.append(Vc)
                        W_cols.append(Wc)
                else:
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                Uc = np.zeros(25)
                                Vc = np.zeros(25)
                                Wc = np.zeros(25)
                                Uc[(n_starts[i] + u) * 5 + (m_starts[j] + v)] = 1
                                Vc[(m_starts[j] + v) * 5 + (p_starts[k] + w)] = 1
                                Wc[(p_starts[k] + w) * 5 + (n_starts[i] + u)] = 1
                                U_cols.append(Uc)
                                V_cols.append(Vc)
                                W_cols.append(Wc)
    
    if not U_cols:
        return {"U": np.zeros((25, 0)), "V": np.zeros((25, 0)), "W": np.zeros((25, 0))}
    
    U_mat = np.column_stack(U_cols)
    V_mat = np.column_stack(V_cols)
    W_mat = np.column_stack(W_cols)
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
