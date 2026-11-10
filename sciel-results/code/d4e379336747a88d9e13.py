import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 590

def kissing_number11() -> np.ndarray:
    """
    Returns a 2-D array of shape (k, 11) with integer entries (int64).
    Uses the set of all integer vectors in Z^11 with exactly two non-zero entries (each +/-1).
    k = C(11, 2) * 2^2 = 55 * 4 = 220.
    Min squared distance is 2, max squared norm is 2.
    Score = 220 / 2 = 110.
    """
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback: minimal valid configuration
        d = 11
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        return points

    d = 11
    # Pre-allocate array for speed
    k = (d * (d - 1) // 2) * 4  # 220
    arr = np.zeros((k, d), dtype=np.int64)
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    arr[idx, i] = si
                    arr[idx, j] = sj
                    idx += 1
    
    return arr
