<<<<<<< SEARCH
    # --- Strategy 1: Sequential and simple structured sets ---
    candidates = []

    # Sequential
    A_seq = list(range(1, 21))
    candidates.append(A_seq)

    # Geometric progression ratio 2
    A_geo = [2**i for i in range(20)]
    candidates.append(A_geo)

    # Arithmetic progressions
    for a in range(1, 10):
        for d in range(1, 6):
            A_ap = [a + i * d for i in range(n)]
            if max(A_ap) <= max_val:
                candidates.append(A_ap)
=======
    # --- Strategy 1: Sequential and simple structured sets ---
    candidates = []

    # Sequential
    A_seq = list(range(1, 21))
    candidates.append(A_seq)

    # Geometric progression ratio 2
    A_geo = [2**i for i in range(20)]
    candidates.append(A_geo)

    # Arithmetic progressions
    for a in range(1, 10):
        for d in range(1, 6):
            A_ap = [a + i * d for i in range(n)]
            if max(A_ap) <= max_val:
                candidates.append(A_ap)

    # --- Strategy 1b: Low/High split sets ---
    # Split into low half (small numbers, dense sums) and high half (geometric, sparse products)
    for low_max in [10, 15, 20, 30]:
        low = list(range(1, low_max + 1))[:10]
        if len(low) < 10:
            continue
        # High half: geometric progression starting above low_max
        high_start = low_max + 1
        for ratio in [2, 3]:
            high = []
            v = high_start
            while len(high) < 10 and v <= max_val:
                high.append(v)
                v *= ratio
            if len(high) >= 10:
                A_split = low + high[:10]
                if len(set(A_split)) == n and max(A_split) <= max_val:
                    candidates.append(sorted(A_split))
        # High half: multiples of a base
        for base in [high_start, high_start + 1, high_start + 2]:
            high = [base * (i + 1) for i in range(10)]
            if max(high) <= max_val:
                A_split = low + high
                if len(set(A_split)) == n:
                    candidates.append(sorted(A_split))
        # High half: powers of 2 scaled
        for scale in [1, 2, 3]:
            high = [scale * (2**i) for i in range(10)]
            if max(high) <= max_val:
                A_split = low + high
                if len(set(A_split)) == n and min(A_split) >= 1:
                    candidates.append(sorted(A_split))
>>>>>>> REPLACE

