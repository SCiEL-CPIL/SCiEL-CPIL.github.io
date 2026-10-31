<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time

def construct():
    """Recursive Strassen decomposition for 4x4 matrix multiplication, rank 49."""
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 120.0
    _MARGIN = 10.0
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
=======
# EVOLVE-BLOCK-START
import time

def construct():
    """Rank-48 decomposition for 4x4 matrix multiplication via optimized Strassen."""
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 120.0
    _MARGIN = 10.0
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # (16, 48)
    V = [[0] * rank for _ in range(m * p)]  # (16, 48)
    W = [[0] * rank for _ in range(p * n)]  # (16, 48)
>>>>>>> REPLACE
