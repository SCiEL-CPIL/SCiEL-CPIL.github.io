"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Precompute: is_res[i] = 1 if i is a non-zero QR mod p, else 0
    # root[r] = smallest x in [1, p-1] such that x^2 mod p == r
    is_res = [0] * (p + 1)  # index 0..p, we use 1..1008
    root = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        is_res[r] = 1
        if root[r] == 0:
            root[r] = x

    # Build prefix sum: prefix[i] = count of QR in {1, ..., i}
    # prefix[0] = 0, prefix[i] = prefix[i-1] + is_res[i]
    prefix = [0] * (p + 1)
    for i in range(1, p):
        prefix[i] = prefix[i - 1] + is_res[i]

    # Find smallest s in [1, 1004] where s..s+4 are all QR
    # prefix[s+5] - prefix[s] == 5 means all 5 values s+1..s+5 are QR
    # But we want s..s+4, so check prefix[s+4] - prefix[s-1] == 5
    # Actually: is_res[s] + is_res[s+1] + ... + is_res[s+4] == 5
    # prefix[s+4] - prefix[s-1] == 5
    for s in range(1, 1005):
        if time.time() > deadline:
            raise RuntimeError('Time limit exceeded')
        if prefix[s + 4] - prefix[s - 1] == 5:
            roots = [root[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
