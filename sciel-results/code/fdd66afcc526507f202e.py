"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

# Precompute modular inverses of 1..10 mod 101
_INV = [0] + [pow(x, -1, 101) for x in range(1, 11)]

# Precompute all subsets of {1,...,10}: map residue -> list of (size, terms_tuple)
# There are 2^10 - 1 = 1023 non-empty subsets.
_SUBSETS_BY_RESIDUE = [[] for _ in range(101)]
for mask in range(1, 1 << 10):
    terms = tuple(x for x in range(1, 11) if (mask >> (x - 1)) & 1)
    r = 0
    for x in terms:
        r += _INV[x]
    r %= 101
    _SUBSETS_BY_RESIDUE[r].append((len(terms), terms))

# For each residue, sort by size ascending so we can pick shortest first
for r in range(101):
    _SUBSETS_BY_RESIDUE[r].sort(key=lambda t: t[0])


def construct():
    deadline = time.time() + 55  # safety margin
    
    # We want to assign to each residue r a subset (denominators) such that:
    # - The subset sums to r mod 101
    # - Denominators within a row are distinct (guaranteed by subset structure)
    # - Denominators may recur across rows
    # - Minimize the largest row length
    
    # Strategy: binary search on max_len L. For each L, try to find an assignment
    # where each residue gets a subset of size <= L.
    # Since we have 101 residues and 1023 subsets, and subsets can be reused across
    # different residues (denominators may recur across rows), this is simply:
    # For each residue r, pick the smallest-size subset that sums to r.
    # The answer is just max over r of min size for residue r.
    
    # Wait - re-reading: "Denominators may recur across different rows, but never within one row."
    # This means the same denominator value can appear in multiple rows. So each row is
    # independently a subset of {1,...,10}. There's no global constraint on how many times
    # a denominator appears across all rows.
    
    # So the problem reduces to: for each residue r in 0..100, find a subset of {1,...,10}
    # whose inverses sum to r mod 101, minimizing the maximum subset size.
    
    # Since subsets are independent, the optimal solution is: for each r, pick the subset
    # of minimum size that sums to r. The answer is max_r (min_size[r]).
    
    # Let's compute min_size for each residue:
    min_size = [10**9] * 101
    best_terms = [None] * 101
    for r in range(101):
        if _SUBSETS_BY_RESIDUE[r]:
            # Already sorted by size
            min_size[r] = _SUBSETS_BY_RESIDUE[r][0][0]
            best_terms[r] = list(_SUBSETS_BY_RESIDUE[r][0][1])
        else:
            # This shouldn't happen since we have 1023 subsets covering 101 residues
            # but let's handle it
            min_size[r] = 10
            best_terms[r] = list(range(1, 11))
    
    # The minimum possible largest row length is max(min_size)
    # But wait - is it possible that we need to use a larger subset for some residue
    # to allow a smaller one for another? No, because there's no cross-row constraint.
    
    # So the optimal answer is simply: for each r, use the shortest subset summing to r.
    max_len = max(min_size)
    
    result = [best_terms[r] for r in range(101)]
    
    # Verify
    for r in range(101):
        s = sum(_INV[x] for x in result[r]) % 101
        assert s == r, f"Residue {r}: got sum {s}"
        assert len(set(result[r])) == len(result[r]), f"Row {r} has duplicate denominators"
        assert all(1 <= x <= 10 for x in result[r]), f"Row {r} has out-of-range denominator"
    
    return result

# EVOLVE-BLOCK-END
