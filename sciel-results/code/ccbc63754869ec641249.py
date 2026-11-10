import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute modular inverses for denominators 1..10 mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)

    # For each residue 0..100, find the shortest non-empty subset of {1..10}
    # whose inverses sum to that residue mod 101.
    # Total non-empty subsets: 2^10 - 1 = 1023, which is tiny.
    best_lists = [None] * 101

    # Enumerate all non-empty subsets via bitmask
    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
        total = 0
        length = 0
        terms = []
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                terms.append(denom)
                total = (total + inv[denom]) % 101
                length += 1

        r = total
        # Keep the shortest representation for residue r
        if best_lists[r] is None or length < len(best_lists[r]):
            best_lists[r] = terms

    # Handle any missing residues (shouldn't happen with 1023 subsets)
    for r in range(101):
        if best_lists[r] is None:
            best_lists[r] = list(range(1, 11))

    # Verify correctness
    for r in range(101):
        row = best_lists[r]
        assert len(row) > 0, f"Row {r} is empty"
        assert len(set(row)) == len(row), f"Row {r} has duplicate denominators"
        assert all(1 <= x <= 10 for x in row), f"Row {r} has out-of-range denominators"
        computed = sum(inv[x] for x in row) % 101
        assert computed == r, f"Row {r}: sum of inverses is {computed}, expected {r}"

    return best_lists
