"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

# Modular inverses of 1..10 mod 101
_INV = [0] + [pow(x, -1, 101) for x in range(1, 11)]

# Build lookup: residue -> list of (size, frozenset_of_denominators)
# Enumerate all 2^10 - 1 = 1023 non-empty subsets
_RESIDUE_MAP = [[] for _ in range(101)]
for mask in range(1, 1 << 10):
    total = 0
    count = 0
    elems = []
    for x in range(1, 11):
        if mask & (1 << (x - 1)):
            total = (total + _INV[x]) % 101
            count += 1
            elems.append(x)
    _RESIDUE_MAP[total].append((count, frozenset(elems)))

# Sort each residue's list by size ascending
for r in range(101):
    _RESIDUE_MAP[r].sort(key=lambda t: t[0])

# Precompute the optimal answer: for each residue, the minimum-size subset
_OPTIMAL = [None] * 101
for r in range(101):
    _OPTIMAL[r] = list(_RESIDUE_MAP[r][0][1])  # smallest subset for residue r

# The minimum possible largest row length
_MAX_LEN = max(len(_OPTIMAL[r]) for r in range(101))


def construct():
    deadline = time.time() + 55.0

    # Since denominators may recur across rows, each row is an independent
    # subset of {1,...,10}. The optimal strategy is to pick, for each residue r,
    # the subset of minimum cardinality whose inverses sum to r mod 101.
    # This is precomputed above.
    result = [list(_OPTIMAL[r]) for r in range(101)]

    # Verify correctness
    for r in range(101):
        s = sum(_INV[x] for x in result[r]) % 101
        if s != r:
            raise AssertionError(f"Residue {r}: got sum {s}")
        if len(set(result[r])) != len(result[r]):
            raise AssertionError(f"Row {r} has duplicates")
        if not all(1 <= x <= 10 for x in result[r]):
            raise AssertionError(f"Row {r} has out-of-range value")

    return result

# EVOLVE-BLOCK-END
