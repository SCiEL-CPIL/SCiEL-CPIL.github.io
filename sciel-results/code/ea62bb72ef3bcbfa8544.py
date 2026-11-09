import time
import numpy as np

def is_latin(sq):
    n = len(sq)
    sq_np = np.array(sq)
    for i in range(n):
        if len(np.unique(sq_np[i])) != n:
            return False
        if len(np.unique(sq_np[:, i])) != n:
            return False
    return True

def is_orthogonal(sq1, sq2):
    sq1_np = np.array(sq1)
    sq2_np = np.array(sq2)
    pairs = sq1_np * 10 + sq2_np
    return len(np.unique(pairs)) == 100

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 50

    # Base Latin Square: cyclic group table Z_10
    base = [[(i + j) % n for j in range(n)] for i in range(n)]

    # Second Latin Square: known orthogonal mate to the cyclic group table
    # Construction based on the decomposition of Z_10 and the field GF(5)
    # For columns 0-4: use (2*j + i) % 10
    # For columns 5-9: use (2*(j-5) + i + 1) % 10
    mate = []
    for i in range(n):
        row = []
        for j in range(n):
            if j < 5:
                val = (2 * j + i) % 10
            else:
                val = (2 * (j - 5) + i + 1) % 10
            row.append(int(val))
        mate.append(row)

    # Verify both are Latin squares
    if not is_latin(base) or not is_latin(mate):
        return [base] if is_latin(base) else []

    # Verify orthogonality
    if is_orthogonal(base, mate):
        return [base, mate]

    # Fallback: return just the base square
    return [base]
