import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Total rank expected: 50
    # 2 blocks of (2x2x7)
    # Each 2x2x7 = 2x2x2 (7) + 2x2x2 (7) + 2x2x3 (11) = 25
    # Total 50
    
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        nonlocal col
        # A is 2x2 at A[a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2]
        # B is 2x2 at B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
        # C is 2x2 at C[b_col_offset:b_col_offset+2, a_row_offset:a_row_offset+2]
        
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 -= M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map A elements to U indices:
        # A00 -> U[a_row_offset*4 + a_col_offset]
        # A01 -> U[a_row_offset*4 + a_col_offset+1]
        # A10 -> U[(a_row_offset+1)*4 + a_col_offset]
        # A11 -> U[(a_row_offset+1)*4 + a_col_offset+1]
        
        # B elements to V indices:
        # B00 -> V[b_row_offset*7 + b_col_offset]
        # B01 -> V[b_row_offset*7 + b_col_offset+1]
        # B10 -> V[(b_row_offset+1)*7 + b_col_offset]
        # B11 -> V[(b_row_offset+1)*7 + b_col_offset+1]
        
        # C elements to W indices:
        # C00 -> W[b_col_offset*2 + a_row_offset]
        # C01 -> W[(b_col_offset+1)*2 + a_row_offset]
        # C10 -> W[b_col_offset*2 + a_row_offset+1]
        # C11 -> W[(b_col_offset+1)*2 + a_row_offset+1]
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[a_row_offset*4 + a_col_offset, col] = 1
        U[(a_row_offset+1)*4 + a_col_offset+1, col] = 1
        V[b_row_offset*7 + b_col_offset, col] = 1
        V[(b_row_offset+1)*7 + b_col_offset+1, col] = 1
        W[b_col_offset*2 + a_row_offset, col] = 1
        W[(b_col_offset+1)*2 + a_row_offset+1, col] = 1
        col += 1
        
        # Term 2: (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1
        U[(a_row_offset+1)*4 + a_col_offset, col] = 1
        U[(a_row_offset+1)*4 + a_col_offset+1, col] = 1
        V[b_row_offset*7 + b_col_offset, col] = 1
        W[b_col_offset*2 + a_row_offset+1, col] = 1
        col += 1
        
        # Term 3: A00(B01-B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01=1
        U[a_row_offset*4 + a_col_offset, col] = 1
        V[b_row_offset*7 + b_col_offset+1, col] = 1
        V[(b_row_offset+1)*7 + b_col_offset+1, col] = -1
        W[(b_col_offset+1)*2 + a_row_offset, col] = 1
        col += 1
        
        # Term 4: A11(B10-B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C10=1, C11=1
        U[(a_row_offset+1)*4 + a_col_offset+1, col] = 1
        V[(b_row_offset+1)*7 + b_col_offset, col] = 1
        V[b_row_offset*7 + b_col_offset, col] = -1
        W[b_col_offset*2 + a_row_offset+1, col] = 1
        W[(b_col_offset+1)*2 + a_row_offset+1, col] = 1
        col += 1
        
        # Term 5: (A00+A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C01=1, C11=-1
        U[a_row_offset*4 + a_col_offset, col] = 1
        U[a_row_offset*4 + a_col_offset+1, col] = 1
        V[(b_row_offset+1)*7 + b_col_offset+1, col] = 1
        W[(b_col_offset+1)*2 + a_row_offset, col] = 1
        W[(b_col_offset+1)*2 + a_row_offset+1, col] = -1
        col += 1
        
        # Term 6: (A10-A00)(B00+B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11=1
        U[(a_row_offset+1)*4 + a_col_offset, col] = 1
        U[a_row_offset*4 + a_col_offset, col] = -1
        V[b_row_offset*7 + b_col_offset, col] = 1
        V[b_row_offset*7 + b_col_offset+1, col] = 1
        W[(b_col_offset+1)*2 + a_row_offset+1, col] = 1
        col += 1
        
        # Term 7: (A01-A11)(B10+B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1
        U[a_row_offset*4 + a_col_offset+1, col] = 1
        U[(a_row_offset+1)*4 + a_col_offset+1, col] = -1
        V[(b_row_offset+1)*7 + b_col_offset, col] = 1
        V[(b_row_offset+1)*7 + b_col_offset+1, col] = 1
        W[b_col_offset*2 + a_row_offset, col] = 1
        col += 1

    # Helper to add a 2x2x1 block (A is 2x2, B is 2x1)
    # C is 2x1
    def add_direct_2x2x1_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        nonlocal col
        # A 2x2, B 2x1 (B has 1 column, so b_col_offset is the only column)
        # C 2x1
        
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # Term 1: A00 B00 -> C00
        U[a_row_offset*4 + a_col_offset, col] = 1
        V[b_row_offset*7 + b_col_offset, col] = 1
        W[b_col_offset*2 + a_row_offset, col] = 1
        col += 1
        
        # Term 2: A01 B10 -> C00
        U[a_row_offset*4 + a_col_offset+1, col] = 1
        V[(b_row_offset+1)*7 + b_col_offset, col] = 1
        W[b_col_offset*2 + a_row_offset, col] = 1
        col += 1
        
        # Term 3: A10 B00 -> C10
        U[(a_row_offset+1)*4 + a_col_offset, col] = 1
        V[b_row_offset*7 + b_col_offset, col] = 1
        W[b_col_offset*2 + a_row_offset+1, col] = 1
        col += 1
        
        # Term 4: A11 B10 -> C10
        U[(a_row_offset+1)*4 + a_col_offset+1, col] = 1
        V[(b_row_offset+1)*7 + b_col_offset, col] = 1
        W[b_col_offset*2 + a_row_offset+1, col] = 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:7]
    # Sub-block 1a: 2x2x2 (cols 0:2 of B)
    add_strassen_block(0, 0, 0, 0)
    # Sub-block 1b: 2x2x2 (cols 2:4 of B)
    add_strassen_block(0, 0, 0, 2)
    # Sub-block 1c: 2x2x3 (cols 4:7 of B)
    # Split 2x2x3 into 2x2x2 (cols 4:6) + 2x2x1 (col 6)
    add_strassen_block(0, 0, 0, 4)
    add_direct_2x2x1_block(0, 0, 0, 6)

    # Block 2: A[0:2, 2:4] * B[2:4, 0:7]
    # Sub-block 2a: 2x2x2 (cols 0:2 of B)
    add_strassen_block(0, 2, 2, 0)
    # Sub-block 2b: 2x2x2 (cols 2:4 of B)
    add_strassen_block(0, 2, 2, 2)
    # Sub-block 2c: 2x2x3 (cols 4:7 of B)
    add_strassen_block(0, 2, 2, 4)
    add_direct_2x2x1_block(0, 2, 2, 6)
    
    # col should be 50
    if col != 50:
        # Fallback to schoolbook if logic error
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
