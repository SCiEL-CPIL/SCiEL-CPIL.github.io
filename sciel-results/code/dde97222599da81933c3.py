"""Combined program: NumPy-accelerated run detection with deadline guard."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill

    p = 1009
    m = 5

    # Precompute: boolean residue array and smallest-root map
    is_res = np.zeros(p, dtype=bool)
    root_map = np.zeros(p, dtype=np.int32)

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if not is_res[r]:
                is_res[r] = True
                root_map[r] = x

    # NumPy sliding-window detection: find first s where s..s+m-1 are all residues
    # is_res[1:p] has length p-1; index j corresponds to value j+1
    res_slice = is_res[1:p]  # length 1008, indices 0..1007 -> values 1..1008

    # Prefix sum for O(1) window sums
    prefix = np.concatenate(([0], np.cumsum(res_slice.astype(np.int32))))
    # window_sums[j] = sum of res_slice[j : j+m] = count of residues in values j+1 .. j+m
    # We need window_sums[j] == m for all m values to be residues
    # Valid j range: 0 to (p-1-m) = 1003, corresponding to s = j+1 in [1, 1004]
    window_sums = prefix[m:] - prefix[:-m]  # length p-1-m+1 = 1004

    valid = np.where(window_sums == m)[0]

    if len(valid) > 0:
        s = int(valid[0]) + 1
        roots = [int(root_map[s + i]) for i in range(m)]
        return {'start': s, 'roots': roots}

    # Fallback: linear scan (should not be reached)
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            raise RuntimeError("Deadline exceeded")
        if all(is_res[s + i] for i in range(m)):
            roots = [int(root_map[s + i]) for i in range(m)]
            return {'start': s, 'roots': roots}

    raise RuntimeError("No consecutive run of quadratic residues found")
# EVOLVE-BLOCK-END
