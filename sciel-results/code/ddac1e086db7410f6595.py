import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0

    # Thue-Morse initialization
    s = [1] * n
    for i in range(1, n):
        if i % 2 == 0:
            s[i] = s[i // 2]
        else:
            s[i] = -s[i // 2]

    # Precompute all APs as lists of indices
    # AP defined by (start, step, length) where start in [0,n), step >= 1, length >= 1
    # We need all maximal APs (can't extend further)
    # For efficiency, group APs by step
    # An AP with step d starting at a has indices a, a+d, a+2d, ...
    # Maximal means a < d (so it can't be extended backwards) and a + k*d < n

    # For incremental evaluation, we track sum of each AP
    # When we flip index i, all APs containing i change by ±2

    # Precompute: for each index i, which APs contain it?
    # This could be large, so let's think differently.

    # Alternative: maintain an array ap_sums where ap_sums[(a,d)] = sum of AP starting at a with step d
    # But there are O(n^2) such APs.

    # Better approach: For each step d, the APs are determined by starting positions a in [0, d-1]
    # (maximal APs). There are d such APs per step, total ~ n log n APs.
    # For each AP, track its sum. When flipping index i, find all APs containing i.

    # For a given index i and step d, the APs with step d containing i are those with start a ≡ i (mod d), a < d.
    # So for each d, there's exactly one maximal AP containing i (if i < n).
    # Wait, that's not right. The maximal APs with step d are starts 0, 1, ..., d-1.
    # Index i is in the AP with start (i % d) and step d.
    # So for each d from 1 to n-1, index i belongs to exactly one maximal AP.
    # Total APs: sum_{d=1}^{n-1} d = n(n-1)/2 ≈ 32768 for n=256.

    # Let's precompute for each index i, the list of (d, start) pairs for maximal APs containing i.
    # For step d, start = i % d. The AP is (start, d).
    # We need start + k*d < n for some k, which is always true if i < n and start = i%d.

    # Actually, we also need to consider that the AP must be non-empty and within bounds.
    # The maximal AP with step d and start a (where a < d) contains indices a, a+d, a+2d, ... < n.
    # Index i is in this AP iff i ≡ a (mod d), i.e., a = i % d.

    # So for each i and each d in [1, n-1], the AP is (i % d, d).
    # Total entries in the mapping: n * (n-1) ≈ 65536.

    # Let's build: for each index i, list of (d, start) for d in 1..n-1
    # And maintain ap_sum[(d, start)] = current sum of that AP

    # For evaluation, we need max |ap_sum| over all (d, start).
    # We can maintain a histogram of ap_sum values.

    # Let's use a different approach: maintain a list of all APs and their sums.
    # Number of APs: sum_{d=1}^{n-1} d = n(n-1)/2 = 32640 for n=256.

    # Build AP list
    aps = []  # list of (d, start, indices)
    ap_index = {}  # (d, start) -> index in aps
    for d in range(1, n):
        for start in range(d):
            indices = list(range(start, n, d))
            if indices:
                ap_index[(d, start)] = len(aps)
                aps.append(indices)

    num_aps = len(aps)

    # For each index i, which APs contain it?
    index_to_aps = [[] for _ in range(n)]
    for d in range(1, n):
        for start in range(d):
            ai = ap_index[(d, start)]
            for idx in aps[ai]:
                index_to_aps[idx].append(ai)

    # Initialize AP sums
    ap_sums = [0] * num_aps
    for i in range(n):
        sign = s[i]
        for ai in index_to_aps[i]:
            ap_sums[ai] += sign

    # Build histogram: count of APs with each sum value
    # Sums range from -n to n, so at most 2n+1 = 513 values
    # Offset by n to make non-negative
    max_sum_range = n
    hist_size = 2 * max_sum_range + 1
    hist = [0] * hist_size

    def update_hist(val, delta):
        """Update histogram: add delta to count of value val"""
        idx = val + max_sum_range
        if 0 <= idx < hist_size:
            hist[idx] += delta

    for ai in range(num_aps):
        update_hist(ap_sums[ai], 1)

    def get_max_disc():
        """Get maximum absolute AP sum from histogram"""
        # Check from edges inward
        for offset in range(max_sum_range, -1, -1):
            if offset == 0:
                if hist[max_sum_range] > 0:
                    return 0
            else:
                if hist[max_sum_range + offset] > 0 or hist[max_sum_range - offset] > 0:
                    return offset
        return 0

    current_disc = get_max_disc()
    best_s = s[:]
    best_disc = current_disc

    rng = random.Random(42)

    def do_flip(i):
        """Flip index i, update AP sums and histogram. Returns old sign."""
        old_sign = s[i]
        new_sign = -old_sign
        delta = 2 * new_sign  # change in sum is 2*new_sign - 2*old_sign = 2*(new_sign - old_sign) = 2*new_sign since new_sign=-old_sign
        # Actually: new_sum = old_sum + (new_sign - old_sign) = old_sum + 2*new_sign
        # Since new_sign = -old_sign, change = -2*old_sign = 2*new_sign
        for ai in index_to_aps[i]:
            old_ap_sum = ap_sums[ai]
            new_ap_sum = old_ap_sum + delta
            update_hist(old_ap_sum, -1)
            ap_sums[ai] = new_ap_sum
            update_hist(new_ap_sum, 1)
        s[i] = new_sign

    def undo_flip(i):
        """Undo a flip at index i (i.e., flip it back)."""
        do_flip(i)

    # Iterated Local Search
    best_disc = current_disc
    best_s = s[:]

    while time.time() < deadline:
        # Local search: keep flipping if it improves
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try random flips, prefer those in high-discrepancy APs
            # Get APs with max absolute sum
            max_d = get_max_disc()
            if max_d == 0:
                break

            # Find APs with |sum| == max_d
            # For simplicity, try random indices
            # To be smarter, pick indices from APs with high |sum|
            # But that's expensive. Let's just try random flips with some bias.

            # Try up to some number of flips
            for _ in range(100):
                if time.time() > deadline:
                    break
                i = rng.randint(0, n - 1)
                do_flip(i)
                new_disc = get_max_disc()
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_s = s[:]
                    # Try to continue improving from here
                    break
                else:
                    undo_flip(i)
                    current_disc = get_max_disc()

        # If no improvement, perturb
        if time.time() > deadline:
            break

        # Perturb: flip 3-5 random positions
        num_perturb = rng.randint(3, 5)
        for _ in range(num_perturb):
            i = rng.randint(0, n - 1)
            do_flip(i)
        current_disc = get_max_disc()

        # If perturbation made things much worse, revert to best
        if current_disc > best_disc + 3:
            # Revert to best
            for i in range(n):
                if s[i] != best_s[i]:
                    do_flip(i)
            current_disc = best_disc

    return best_s
