"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct exactly 12 distinct positive integers <= 1048576 such that
    all 4096 subset sums are distinct. Minimize the largest integer.
    
    Theoretical bound: for n elements with distinct subset sums, if sorted
    a_1 < ... < a_n, then a_k > sum_{i<k} a_i, forcing a_k >= 2^(k-1).
    Thus a_12 >= 2048. Powers of 2 achieve this bound exactly.
    
    We use a greedy construction that picks the smallest valid candidate at
    each step, verified via subset-sum set disjointness check.
    """
    n = 12
    max_val = 1048576
    start = time.time()
    deadline = start + 80  # 80s margin within 90s limit

    seq = []
    sums = {0}

    for i in range(n):
        if time.time() > deadline:
            break

        candidate = 1
        found = False
        while candidate <= max_val:
            if time.time() > deadline:
                break

            # Check: for all s in sums, (s + candidate) not in sums
            # This ensures the new subset sums don't collide with old ones.
            valid = True
            for s in sums:
                if s + candidate in sums:
                    valid = False
                    break

            if valid:
                seq.append(candidate)
                # Update sums: old sums plus (old sums + candidate)
                old_sums = sums
                sums = set(old_sums)
                for s in old_sums:
                    sums.add(s + candidate)
                found = True
                break

            candidate += 1

        if not found:
            break

    if len(seq) == n:
        return seq

    # Fallback: powers of 2 (provably optimal, largest = 2048)
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
