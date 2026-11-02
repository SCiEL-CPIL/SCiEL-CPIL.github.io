import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = DIMENSIONS
    
    # Current known valid rank-46 solution
    rank46 = _build_rank46()
    
    # Try to improve with border rank trick
    # The idea: use 7 Strassen terms for the first 4 columns,
    # then use fewer terms for remaining columns by exploiting linearity
    
    # Try rank 45: can we merge some of the schoolbook terms?
    rank45 = _try_rank45()
    if rank45 is not None and time.time() < deadline - 30:
        return rank45
    
    # Try rank 44
    rank44 = _try_rank44()
    if rank44 is not None and time.time() < deadline - 20:
        return rank44
    
    # Try rank 43
    rank43 = _try_rank43()
    if rank43 is not None and time.time() < deadline - 15:
        return rank43
    
    # Try rank 42
    rank42 = _try_rank42()
    if rank42 is not None and time.time() < deadline - 10:
        return rank42
    
    # Try rank 41 (border rank trick)
    rank41 = _try_rank41()
    if rank41 is not None and time.time() < deadline - 5:
        return rank41
    
    # Fallback to rank 46
    return rank46


def _build_rank46():
    n, m, p = DIMENSIONS
    RANK = 46
    u_rows = n * m       # 4
    v_rows = m * p       # 26
    w_rows = p * n       # 26

    U = [[0] * RANK for _ in range(u_rows)]
    V = [[0] * RANK for _ in range(v_rows)]
    W = [[0] * RANK for _ in range(w_rows)]

    offset = 0

    def emit(u_map, v_map, w_map):
        nonlocal offset
        for idx, val in u_map.items():
            U[idx][offset] += val
        for idx, val in v_map.items():
            V[idx][offset] += val
        for idx, val in w_map.items():
            W[idx][offset] += val
        offset += 1

    # Six 2x2x2 Strassen blocks
    for blk in range(6):
        a = blk * 2
        b = blk * 2 + 1

        # M1 = (A00+A11)*(B0a+B1b)
        emit({0: 1, 3: 1}, {a: 1, 13 + b: 1}, {2*a: 1, 2*b + 1: 1})
        # M2 = (A10+A11)*B0a
        emit({2: 1, 3: 1}, {a: 1}, {2*a + 1: 1, 2*b + 1: -1})
        # M3 = A00*(B1b-B0b)
        emit({0: 1}, {b: 1, 13 + b: -1}, {2*b: 1, 2*b + 1: 1})
        # M4 = A11*(B1a-B0a)
        emit({3: 1}, {13 + a: 1, a: -1}, {2*a: 1, 2*a + 1: 1})
        # M5 = (A00+A01)*B1b
        emit({0: 1, 1: 1}, {13 + b: 1}, {2*a: -1, 2*b: 1})
        # M6 = (A10-A00)*(B0a+B0b)
        emit({2: 1, 0: -1}, {a: 1, b: 1}, {2*b + 1: 1})
        # M7 = (A01-A11)*(B1a+B1b)
        emit({1: 1, 3: -1}, {13 + a: 1, 13 + b: 1}, {2*a: 1})

    # Final 2x2x1 schoolbook block for column k=12
    k = 12
    emit({0: 1}, {k: 1}, {2*k: 1})
    emit({1: 1}, {13 + k: 1}, {2*k: 1})
    emit({2: 1}, {k: 1}, {2*k + 1: 1})
    emit({3: 1}, {13 + k: 1}, {2*k + 1: 1})

    return {"U": U, "V": V, "W": W}


def _try_rank45():
    # Try to reduce rank by 1 by finding a dependency
    # This requires solving a system of linear equations
    # For now, return None as we don't have a guaranteed method
    return None


def _try_rank44():
    return None


def _try_rank43():
    return None


def _try_rank42():
    return None


def _try_rank41():
    # Border rank trick: use 7 terms for 4 columns, then 9*4=36 for rest
    # But we need to be smarter about the remaining 9 columns
    # The key insight: for each column k, the 2x2x1 multiplication has rank 4
    # But if we can share structure between columns, we might reduce it
    
    # Actually, the border rank approach suggests that for the last few columns,
    # we might be able to use fewer than 4 terms each by exploiting the fact
    # that they are "border" terms
    
    # For now, return None as implementing this correctly is complex
    return None


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
