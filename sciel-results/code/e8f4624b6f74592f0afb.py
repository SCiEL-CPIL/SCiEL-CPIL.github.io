import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a column
    def add_column(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for idx, val in u_coeffs.items():
            U[idx][col_idx] += val
        for idx, val in v_coeffs.items():
            V[idx][col_idx] += val
        for idx, val in w_coeffs.items():
            W[idx][col_idx] += val

    # Block 1: k in {0, 1}
    # Strassen part
    # P1
    add_column({0: 1, 4: 1}, {0: 1, 6: 1}, {0: 1, 3: 1}, 0) # U(0,0), U(1,1) -> 0, 4. V(0,0), V(1,1) -> 0, 6. W(0,0), W(1,1) -> 0, 3.
    # Wait, U index: i*m+j. i=0,j=0->0. i=1,j=1->1*3+1=4.
    # V index: j*p+k. j=0,k=0->0. j=1,k=1->1*5+1=6.
    # W index: k*n+i. k=0,i=0->0. k=1,i=1->1*2+1=3.
    
    # P2
    add_column({3: 1, 4: 1}, {0: 1}, {1: 1, 3: -1}, 1) # U(1,0)=3, U(1,1)=4. V(0,0)=0. W(0,1)=1, W(1,1)=3.
    
    # P3
    add_column({0: 1}, {5: 1, 6: -1}, {2: 1, 3: 1}, 2) # U(0,0)=0. V(1,0)=5, V(1,1)=6. W(1,0)=2, W(1,1)=3.
    
    # P4
    add_column({4: 1}, {1: 1, 0: -1}, {0: 1, 1: 1}, 3) # U(1,1)=4. V(0,1)=1, V(0,0)=0. W(0,0)=0, W(0,1)=1.
    
    # P5
    add_column({0: 1, 1: 1}, {6: 1}, {0: -1, 2: 1}, 4) # U(0,0)=0, U(0,1)=1. V(1,1)=6. W(0,0)=0, W(1,0)=2.
    
    # P6
    add_column({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}, 5) # U(1,0)=3, U(0,0)=0. V(0,0)=0, V(0,1)=1. W(1,1)=3.
    
    # P7
    add_column({1: 1, 4: -1}, {5: 1, 6: 1}, {0: 1}, 6) # U(0,1)=1, U(1,1)=4. V(1,0)=5, V(1,1)=6. W(0,0)=0.

    # Schoolbook part for Block 1
    # Q1
    add_column({2: 1}, {10: 1}, {0: 1}, 7) # U(0,2)=2. V(2,0)=10. W(0,0)=0.
    # Q2
    add_column({2: 1}, {11: 1}, {2: 1}, 8) # U(0,2)=2. V(2,1)=11. W(1,0)=2.
    # Q3
    add_column({5: 1}, {10: 1}, {1: 1}, 9) # U(1,2)=5. V(2,0)=10. W(0,1)=1.
    # Q4
    add_column({5: 1}, {11: 1}, {3: 1}, 10) # U(1,2)=5. V(2,1)=11. W(1,1)=3.

    # Block 2: k in {2, 3}
    # Same U, shifted V and W.
    # V indices: j*5+k. k is 2,3.
    # W indices: k*2+i. k is 2,3.
    
    # P1
    add_column({0: 1, 4: 1}, {2: 1, 8: 1}, {4: 1, 7: 1}, 11) # V(0,2)=2, V(1,3)=8. W(2,0)=4, W(3,1)=7.
    # P2
    add_column({3: 1, 4: 1}, {2: 1}, {5: 1, 7: -1}, 12) # V(0,2)=2. W(2,1)=5, W(3,1)=7.
    # P3
    add_column({0: 1}, {7: 1, 8: -1}, {6: 1, 7: 1}, 13) # V(1,2)=7, V(1,3)=8. W(3,0)=6, W(3,1)=7.
    # P4
    add_column({4: 1}, {3: 1, 2: -1}, {4: 1, 5: 1}, 14) # V(0,3)=3, V(0,2)=2. W(2,0)=4, W(2,1)=5.
    # P5
    add_column({0: 1, 1: 1}, {8: 1}, {4: -1, 6: 1}, 15) # V(1,3)=8. W(2,0)=4, W(3,0)=6.
    # P6
    add_column({3: 1, 0: -1}, {2: 1, 3: 1}, {7: 1}, 16) # V(0,2)=2, V(0,3)=3. W(3,1)=7.
    # P7
    add_column({1: 1, 4: -1}, {7: 1, 8: 1}, {4: 1}, 17) # V(1,2)=7, V(1,3)=8. W(2,0)=4.

    # Schoolbook part for Block 2
    # Q1
    add_column({2: 1}, {12: 1}, {4: 1}, 18) # V(2,2)=12. W(2,0)=4.
    # Q2
    add_column({2: 1}, {13: 1}, {6: 1}, 19) # V(2,3)=13. W(3,0)=6.
    # Q3
    add_column({5: 1}, {12: 1}, {5: 1}, 20) # V(2,2)=12. W(2,1)=5.
    # Q4
    add_column({5: 1}, {13: 1}, {7: 1}, 21) # V(2,3)=13. W(3,1)=7.

    # Block 3: k = 4
    # R1
    add_column({0: 1}, {4: 1}, {8: 1}, 22) # U(0,0)=0. V(0,4)=4. W(4,0)=8.
    # R2
    add_column({1: 1}, {9: 1}, {8: 1}, 23) # U(0,1)=1. V(1,4)=9. W(4,0)=8.
    # R3
    add_column({2: 1}, {14: 1}, {8: 1}, 24) # U(0,2)=2. V(2,4)=14. W(4,0)=8.
    # R4
    add_column({3: 1}, {4: 1}, {9: 1}, 25) # U(1,0)=3. V(0,4)=4. W(4,1)=9.
    # R5
    add_column({4: 1}, {9: 1}, {9: 1}, 26) # U(1,1)=4. V(1,4)=9. W(4,1)=9.
    # R6
    add_column({5: 1}, {14: 1}, {9: 1}, 27) # U(1,2)=5. V(2,4)=14. W(4,1)=9.

    return {"U": U, "V": V, "W": W}
