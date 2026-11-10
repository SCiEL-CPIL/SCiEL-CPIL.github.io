import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    p = 1009

    # Precompute quadratic residues using array for O(1) lookup
    is_residue = [False] * p
    root_of = [0] * p

    for r in range(1, p):
        res = (r * r) % p
        if not is_residue[res]:
            is_residue[res] = True
            root_of[res] = r

    # Search for the smallest s in [1, 1004] such that
    # s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    for s in range(1, 1005):
        if time.time() - start_time > deadline:
            break

        if (is_residue[s] and is_residue[s + 1] and is_residue[s + 2]
                and is_residue[s + 3] and is_residue[s + 4]):
            return {
                'start': s,
                'roots': [root_of[s], root_of[s + 1], root_of[s + 2],
                          root_of[s + 3], root_of[s + 4]]
            }

    # Fallback (should not be reached for p=1009, m=5)
    raise RuntimeError("No valid run found")
