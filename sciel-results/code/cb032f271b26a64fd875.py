# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q = 49, p = 7, alpha^2 = 3).
    N is a Nikodym set if for every point x in F_q^2 (in N or not) there is a
    direction v = (1, m) for m in F_49, or v = (0,1), such that the punctured
    line {x + t*v : t != 0} is fully contained in N. We maximize |complement|.

    Design (baseline family, refined):
      - Bitmask representation of N for O(1) line-containment checks.
      - Start from the known-good baseline (exclude non-QR-y unital points).
      - Verify-and-revert greedy: exclude extra unital points only while the
        full Nikodym property is preserved (verified over all 2401 points).
    """
    p = 7
    q = 49
    K = 3

    def idx(xa, xb, ya, yb):
        return xa + xb * p + ya * p * p + yb * p * p * p

    def decode(pt):
        return (pt % p, (pt // p) % p, (pt // (p * p)) % p, (pt // (p * p * p)) % p)

    # Hermitian unital U: 2*x_a + y_a^2 - K*y_b^2 = 0 (mod p), x_b free.
    unital = set()
    for xa in range(p):
        for ya in range(p):
            for yb in range(p):
                xb = (-2 * xa - ya * ya + yb * yb * K) % p
                unital.add(idx(xa, xb, ya, yb))
    unital_list = sorted(unital)

    QR = set((u * u) % p for u in range(p))
    S_pts = []  # non-QR y unital -> baseline exclusions
    B_pts = []  # QR y unital -> candidates for extra exclusion
    for pt in unital_list:
        xa, xb, ya, yb = decode(pt)
        if ya in QR:
            B_pts.append(pt)
        else:
            S_pts.append(pt)

    all_mask = (1 << q) - 1

    # Precompute, for every point, the bitmask of the other 48 points on each
    # of the 50 witness lines (49 slanted v=(1,m) + 1 vertical v=(0,1)).
    all_point_lines = []
    for pt in range(q):
        xa, xb, ya, yb = decode(pt)
        lines = []
        for m in range(q):
            ma, mb = m // p, m % p
            mask = 0
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    npt = idx((xa + ta) % p, (xb + tb) % p,
                              (ya + ta * ma) % p, (yb + ta * mb) % p)
                    mask |= (1 << npt)
            lines.append(mask)
        mask = 0
        for ta in range(p):
            for tb in range(p):
                if ta == 0 and tb == 0:
                    continue
                npt = idx(xa, xb, (ya + tb) % p, yb)
                mask |= (1 << npt)
        lines.append(mask)
        all_point_lines.append(lines)

    def is_valid(N_mask):
        apl = all_point_lines
        for pt in range(q):
            for lmask in apl[pt]:
                if (N_mask & lmask) == lmask:
                    break
            else:
                return False
        return True

    # Baseline N (known valid).
    N = all_mask
    for pt in S_pts:
        N &= ~(1 << pt)
    if not is_valid(N):
        N = all_mask  # safety fallback

    # Verify-and-revert greedy to exclude extra unital points.
    start = time.time()
    DEADLINE = 590.0
    progress = True
    while progress and time.time() < DEADLINE:
        progress = False
        for pt in B_pts:
            if time.time() > DEADLINE:
                break
            if (N >> pt) & 1:  # still in N
                N_tent = N & ~(1 << pt)
                if is_valid(N_tent):
                    N = N_tent
                    progress = True

    points = []
    for pt in range(q):
        if (N >> pt) & 1:
            xa, xb, ya, yb = decode(pt)
            points.append([[xa, xb], [ya, yb]])

    if len(points) == 0:
        for pt in range(q):
            xa, xb, ya, yb = decode(pt)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
