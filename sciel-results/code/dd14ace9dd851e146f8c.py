import time

def construct():
    deadline = time.time() + 85  # safety margin within 90s limit

    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]

    # For each residue, find the minimum-size subset that maps to it
    # Initialize with infinity
    min_size = [float('inf')] * 101
    best_subset = [None] * 101

    # Enumerate all non-empty subsets of {1..10}
    # Group by size, process small sizes first
    for size in range(1, 11):
        if time.time() > deadline:
            break
        # Enumerate all subsets of exactly this size
        from itertools import combinations
        for combo in combinations(range(1, 11), size):
            residue = sum(inv[x - 1] for x in combo) % 101
            if size < min_size[residue]:
                min_size[residue] = size
                best_subset[residue] = list(combo)

    # All residues should be covered (there are 1023 subsets for 101 residues)
    # Build result
    result = []
    for r in range(101):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # Fallback: find any subset (shouldn't happen)
            found = False
            for size in range(1, 11):
                from itertools import combinations
                for combo in combinations(range(1, 11), size):
                    if sum(inv[x - 1] for x in combo) % 101 == r:
                        result.append(list(combo))
                        found = True
                        break
                if found:
                    break
            if not found:
                result.append([1])  # ultimate fallback

    return result
