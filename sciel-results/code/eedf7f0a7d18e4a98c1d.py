"""Combined program: MODEL-AND-OPTIMIZE with precomputation.
We model the answer as an assignment of one subset (denominator mask) to each residue r mod 101.
Objective: minimize the maximum row length (i.e., the max popcount over all 101 rows).
We precompute all 1023 nonempty masks with their residues and popcounts.
Then we find the minimum feasible max length k such that every residue can be covered
by at least one mask of popcount <= k, and assign the smallest-popcount mask per residue.

Since 101 residues and 10 denominators, the number of distinct subsets is 1023.
We can precompute for each residue the list of masks (by popcount ascending) that achieve it.
Then for a given k, we check if every residue has at least one mask with popcount <= k.
The minimum such k is the optimal max row length. Then we just pick one such mask per residue.
"""
import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill

    MOD = 101
    n = 10

    # Precompute modular inverses of 1..10 mod 101
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, -1, MOD)
        
    # Precompute sums and sizes for all masks to speed up the loop slightly
    mask_sums = [0] * (1 << n)
    mask_sizes = [0] * (1 << n)
    for mask in range(1, 1 << n):
        # Use least significant set bit trick or simple loop
        # Simple loop is fine for 1023 items
        pass 

    # For each residue, store the best (smallest popcount) mask
    # min_size_for_res[r] = minimum popcount of any mask summing to r
    # best_mask_for_res[r] = the mask achieving that minimum
    min_size_for_res = [11] * MOD
    best_mask_for_res = [0] * MOD

    # Iterate through all non-empty subsets of {1..10}
    # Optimization: Precompute contribution of each bit
    # But n=10 is small, so direct calculation is fast enough.
    for mask in range(1, 1 << n):
        s = 0
        sz = 0
        for x in range(1, n + 1):
            if mask & (1 << (x - 1)):
                s = (s + inv[x]) % MOD
                sz += 1

        r = s
        if sz < min_size_for_res[r]:
            min_size_for_res[r] = sz
            best_mask_for_res[r] = mask

    # Convert masks to lists of denominators
    def mask_to_list(mask):
        return [x for x in range(1, n + 1) if mask & (1 << (x - 1))]

    result = []
    for r in range(MOD):
        m = best_mask_for_res[r]
        row = mask_to_list(m)
        result.append(row)

    # Verify
    for r, terms in enumerate(result):
        if len(terms) == 0:
            raise ValueError(f"Row {r} is empty")
        if len(set(terms)) != len(terms):
            raise ValueError(f"Row {r} has duplicate denominators")
        if any(x < 1 or x > 10 for x in terms):
            raise ValueError(f"Row {r} has invalid denominator")
        res = sum(inv[x] for x in terms) % MOD
        if res != r:
            raise ValueError(f"Row {r} sums to {res} instead of {r}")

    return result
