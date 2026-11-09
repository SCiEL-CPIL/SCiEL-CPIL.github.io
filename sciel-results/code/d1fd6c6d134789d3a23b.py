# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 590  # Safety margin


def kissing_number11() -> np.ndarray:
    """
    Returns a 2-D array of shape (k, 11) with integer entries (int64) such that:
      - second dimension is 11
      - k >= 2
      - no point is the zero vector
      - min squared pairwise distance >= max squared norm (i.e., points are 'kissing' in the strong sense)
    
    Strategy: use all integer vectors in Z^11 with exactly two non-zero entries, each +/-1.
    Count = C(11,2) * 2^2 = 55 * 4 = 220.
    
    Validity check (verified analytically):
      - Each vector has squared norm = 2.
      - For any two distinct such vectors u, v, |u - v|^2 = ||u||^2 + ||v||^2 - 2 u·v = 4 - 2 u·v.
      - u·v is an integer in {-2, -1, 0, 1, 2}.
      - Minimum |u - v|^2 occurs when u·v is maximum.
      - For two distinct weight-2 vectors:
        - If they share 2 positions (same support): u·v in {-2, 0} (since u≠v, can't be 2).
          |u-v|^2 = 4 - 2*(0) = 4 or 4 - 2*(-2) = 8.
        - If they share 1 position: u·v in {-1, 1}. |u-v|^2 in {2, 6}.
        - If they share 0 positions: u·v = 0. |u-v|^2 = 4.
      - So overall min |u-v|^2 = 2, max ||u||^2 = 2. 2 >= 2 holds. ✓
    
    So k = 220 is valid.
    """
    # Check time budget
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback: return a minimal valid configuration
        d = 11
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        return points
    
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    arr = np.array(points, dtype=np.int64)
    arr = np.around(arr).astype(np.int64)
    return arr


# EVOLVE-BLOCK-END
