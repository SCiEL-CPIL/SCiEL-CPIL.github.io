"""
Solves for the earliest run of 5 consecutive quadratic residues modulo 1009.
Returns the start index and the corresponding smallest positive square roots.
"""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    p = 1009
    m = 5

    # Precompute: for each non-zero quadratic residue, store its smallest positive root.
    # x and p-x give the same square, so we only need x in 1..(p-1)//2 to get the
    # smallest root. But scanning 1..p-1 is cheap (1008 iterations) and guarantees
    # the first hit is the smallest root.
    root_map = {}
    for x in range(1, (p + 1) // 2 + 1):
        r = (x * x) % p
        if r != 0:
            # x <= (p-1)//2 ensures this is the smallest positive root for r
            # (since p-x > (p-1)//2 for x < p/2)
            if r not in root_map:
                root_map[r] = x

    # Find smallest s in [1, p-m] such that s, s+1, ..., s+m-1 are all in root_map.
    # p-m = 1004, so s+m-1 <= 1008 (no wraparound, no zero residue).
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            raise RuntimeError("Deadline exceeded")
        # Check all m consecutive values
        all_res = True
        for i in range(m):
            if (s + i) not in root_map:
                all_res = False
                break
        if all_res:
            roots = [root_map[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    raise RuntimeError("No consecutive run of quadratic residues found")
