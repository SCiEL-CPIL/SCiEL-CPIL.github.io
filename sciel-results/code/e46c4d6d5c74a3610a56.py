import time
import sys

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Precompute all column assignments for all blocks
    # Each entry: (matrix, row, col, value)
    assignments = []
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            
            if s == 0:
                assignments.append(('U', 0, col, 1))
                assignments.append(('U', 3, col, 1))
                assignments.append(('V', 0, col, 1))
                assignments.append(('V', p - 1, col, 1))
                assignments.append(('W', 0, col, 1))
                assignments.append(('W', n * p - 1, col, 1))
            elif s == 1:
                assignments.append(('U', 2, col, 1))
                assignments.append(('U', 3, col, 1))
                assignments.append(('V', 0, col, 1))
                assignments.append(('W', 2 * n, col, 1))
                assignments.append(('W', 3 * n, col, -1))
            elif s == 2:
                assignments.append(('U', 0, col, 1))
                assignments.append(('V', n, col, 1))
                assignments.append(('V', p + n - 1, col, -1))
                assignments.append(('W', n, col, 1))
                assignments.append(('W', 3 * n, col, 1))
            elif s == 3:
                assignments.append(('U', 3, col, 1))
                assignments.append(('V', p - 1, col, 1))
                assignments.append(('V', p - 2, col, -1))
                assignments.append(('W', 0, col, 1))
                assignments.append(('W', 2 * n, col, 1))
            elif s == 4:
                assignments.append(('U', 0, col, 1))
                assignments.append(('U', 1, col, 1))
                assignments.append(('V', p + n - 1, col, 1))
                assignments.append(('W', 0, col, -1))
                assignments.append(('W', n, col, 1))
            elif s == 5:
                assignments.append(('U', 2, col, 1))
                assignments.append(('U', 0, col, -1))
                assignments.append(('V', 0, col, 1))
                assignments.append(('V', n, col, 1))
                assignments.append(('W', 3 * n, col, 1))
            elif s == 6:
                assignments.append(('U', 1, col, 1))
                assignments.append(('U', 3, col, -1))
                assignments.append(('V', p - 1, col, 1))
                assignments.append(('V', p + n - 1, col, 1))
                assignments.append(('W', 0, col, 1))
    
    # Apply all assignments
    for mat_name, row, col, val in assignments:
        if time.time() > deadline:
            break
        if mat_name == 'U':
            U[row][col] = val
        elif mat_name == 'V':
            V[row][col] = val
        else:
            W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
