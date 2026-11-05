"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    N = 1008
    MOD = 1009

    # Precompute modular inverses for 1..1008
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)  # Fermat's little theorem, MOD is prime

    best = [2, 1]  # fallback: [2,1] is valid (2*2=4 != 1)

    # For each start, extend end as far as possible
    for start in range(1, N + 1):
        if time.time() > deadline - 3:
            break
        # We'll track the set of elements in current interval
        # Condition: no element x in interval has inv[x] in interval, and x not in {1, 1008}
        elements = set()
        max_end = start - 1
        for end in range(start, N + 1):
            x = end
            # Check diagonal: x must not be 1 or 1008
            if x == 1 or x == 1008:
                break
            # Check if x's inverse is already in the interval
            if inv[x] in elements:
                break
            elements.add(x)
            max_end = end
            if max_end - start + 1 > best[1]:
                best = [start, max_end - start + 1]

    return best
# EVOLVE-BLOCK-END
