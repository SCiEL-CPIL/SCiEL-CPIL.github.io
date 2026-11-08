"""Matrix multiplication 3,3,3 - Rank 26 Decomposition (Refactored)."""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank 26 decomposition using 2x2 and 1x1 block partitioning
    # Strassen for 2x2 (Rank 7) + Schoolbook for the rest
    
    rank = 26
    # U: 9x26, V: 9x26, W: 9x26
    U = [[0.0 for _ in range(rank)] for _ in range(n*m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m*p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p*n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal rank
        # We assume we are filling columns sequentially in the logic below, 
        # but here we need a column index. Let's use a list to track col.
        pass # Placeholder, see below

    # We will use a list of tuples (u_indices, v_indices, w_indices, coeffs)
    # But it's easier to just fill the matrices directly in a loop structure.
    
    col = 0
    
    def term(u_map, v_map, w_map):
        """
        u_map: dict {u_idx: val}
        v_map: dict {v_idx: val}
        w_map: dict {w_idx: val}
        """
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_map.items():
            U[idx][col] = val
        for idx, val in v_map.items():
            V[idx][col] = val
        for idx, val in w_map.items():
            W[idx][col] = val
        col += 1

    # Indices mapping:
    # A: 0 1 2
    #   3 4 5
    #   6 7 8
    # B: 0 1 2
    #   3 4 5
    #   6 7 8
    # C: 0 3 6
    #   1 4 7
    #   2 5 8
    # (W[k*n+i] -> C row k, col i. 
    #  C[0,0]=0, C[0,1]=3, C[0,2]=6
    #  C[1,0]=1, C[1,1]=4, C[1,2]=7
    #  C[2,0]=2, C[2,1]=5, C[2,2]=8)
    
    # 1. Strassen on Top-Left 2x2 (A[0:2,0:2], B[0:2,0:2] -> C[0:2,0:2])
    # A2 indices: 0,1,3,4
    # B2 indices: 0,1,3,4
    # C22 indices: 0,3,1,4
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    term({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1})
    # M2 = (A10+A11)B00 -> C10, C11 (C10 +=, C11 -=)
    term({3: 1, 4: 1}, {0: 1}, {1: 1, 4: -1})
    # M3 = A00(B01-B11) -> C01, C11 (C01 +=, C11 +=)
    term({0: 1}, {1: 1, 4: -1}, {3: 1, 4: 1})
    # M4 = A11(B10-B00) -> C00, C01 (C00 +=, C01 +=)
    term({4: 1}, {3: 1, 0: -1}, {0: 1, 1: 1})
    # M5 = (A00+A01)B11 -> C00, C01 (C00 -=, C01 +=)
    term({0: 1, 1: 1}, {4: 1}, {0: -1, 3: 1})
    # M6 = (A10-A00)(B00+B01) -> C11
    term({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1})
    # M7 = (A01-A11)(B10+B11) -> C00
    term({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1})
    
    # 2. Top-Right (2x1) * Bottom-Left (1x2) -> C Top-Left 2x2
    # A[0:2, 2] -> 2, 5
    # B[2, 0:2] -> 6, 7
    # C[0,0] += A02 B20 (2, 6 -> 0)
    term({2: 1}, {6: 1}, {0: 1})
    # C[0,1] += A02 B21 (2, 7 -> 3)
    term({2: 1}, {7: 1}, {3: 1})
    # C[1,0] += A12 B20 (5, 6 -> 1)
    term({5: 1}, {6: 1}, {1: 1})
    # C[1,1] += A12 B21 (5, 7 -> 4)
    term({5: 1}, {7: 1}, {4: 1})
    
    # 3. Top-Left (2x2) * Top-Right (2x1) -> C Top-Right 2x1
    # A[0:2, 0:2] -> 0,1,3,4
    # B[0:2, 2] -> 2, 5
    # C[0,2] += A00 B02 + A01 B12 (0,2 -> 6; 1,5 -> 6)
    term({0: 1}, {2: 1}, {6: 1})
    term({1: 1}, {5: 1}, {6: 1})
    # C[1,2] += A10 B02 + A11 B12 (3,2 -> 7; 4,5 -> 7)
    term({3: 1}, {2: 1}, {7: 1})
    term({4: 1}, {5: 1}, {7: 1})
    
    # 4. Top-Right (2x1) * Bottom-Right (1x1) -> C Bottom-Right 2x1 part
    # A[0:2, 2] -> 2, 5
    # B[2, 2] -> 8
    # C[0,2] += A02 B22 (2, 8 -> 6)
    term({2: 1}, {8: 1}, {6: 1})
    # C[1,2] += A12 B22 (5, 8 -> 7)
    term({5: 1}, {8: 1}, {7: 1})
    
    # 5. Bottom-Left (1x2) * Top-Left (2x2) -> C Bottom-Left 1x2
    # A[2, 0:2] -> 6, 7
    # B[0:2, 0:2] -> 0,1,3,4
    # C[2,0] += A20 B00 + A21 B10 (6,0 -> 2; 7,3 -> 2)
    term({6: 1}, {0: 1}, {2: 1})
    term({7: 1}, {3: 1}, {2: 1})
    # C[2,1] += A20 B01 + A21 B11 (6,1 -> 5; 7,4 -> 5)
    term({6: 1}, {1: 1}, {5: 1})
    term({7: 1}, {4: 1}, {5: 1})
    
    # 6. Bottom-Right (1x1) * Bottom-Left (1x2) -> C Bottom-Left 1x2 part
    # A[2, 2] -> 8
    # B[2, 0:2] -> 6, 7
    # C[2,0] += A22 B20 (8, 6 -> 2)
    term({8: 1}, {6: 1}, {2: 1})
    # C[2,1] += A22 B21 (8, 7 -> 5)
    term({8: 1}, {7: 1}, {5: 1})
    
    # 7. Bottom-Left (1x2) * Top-Right (2x1) -> C Bottom-Right 1x1
    # A[2, 0:2] -> 6, 7
    # B[0:2, 2] -> 2, 5
    # C[2,2] += A20 B02 + A21 B12 (6,2 -> 8; 7,5 -> 8)
    term({6: 1}, {2: 1}, {8: 1})
    term({7: 1}, {5: 1}, {8: 1})
    
    # 8. Bottom-Right (1x1) * Bottom-Right (1x1) -> C Bottom-Right 1x1
    # A[2, 2] -> 8
    # B[2, 2] -> 8
    # C[2,2] += A22 B22 (8, 8 -> 8)
    term({8: 1}, {8: 1}, {8: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
