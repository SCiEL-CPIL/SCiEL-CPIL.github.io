"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a binary error-correcting code of length 18 and minimum distance 6.
    A_2(18, 6) = 512. We use a time-limited greedy search with structured
    candidate generation and vectorized distance checks via NumPy.
    """
    import numpy as np
    import time

    N = 18
    MIN_DIST = 6
    TARGET = 512
    DEADLINE = time.time() + 55  # Leave 5s buffer

    def popcount(x):
        return x.bit_count()

    # Step 1: Start with 0 and the RM(1,4) code extended to 18 bits
    # RM(1,4) has 32 words of length 16, distance 8
    words = [0]
    for slope in range(16):
        for intercept in range(2):
            if slope == 0 and intercept == 0:
                continue
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Check distance to all existing words
            ok = True
            for w in words:
                if popcount(word ^ w) < MIN_DIST:
                    ok = False
                    break
            if ok:
                words.append(word)

    # Step 2: Greedy search with structured candidate ordering
    # Generate candidates in order of increasing weight to find low-weight words first
    # Then use numpy for batch distance checks

    words_np = np.array(words, dtype=np.uint32)

    # Generate all candidates in a structured way: group by weight
    # For efficiency, we'll iterate through candidates and use numpy for batch checks
    # Process in chunks for efficiency

    n_words = len(words)

    # Try adding candidates in a smart order: by weight, then by value
    # Precompute weight for candidates up to a certain point
    # We'll do multiple passes with different orderings

    # Pass 1: Try all candidates in increasing order of popcount
    # For each candidate, check against all existing words using numpy

    # To speed up, we'll process candidates in batches
    # But first, let's try a simple sequential approach with early termination

    # Candidate generation: iterate through all 2^18 values
    # Use a list to track which have been tried
    tried = np.zeros(1 << N, dtype=bool)

    # We'll do several passes with different candidate orderings
    orderings = [
        "weight_asc",   # increasing popcount
        "weight_desc",  # decreasing popcount
        "value_asc",    # increasing numeric value
        "value_desc",   # decreasing numeric value
    ]

    for ordering in orderings:
        if len(words) >= TARGET:
            break
        if time.time() > DEADLINE:
            break

        # Generate candidate list in the specified order
        all_candidates = list(range(1, 1 << N))

        if ordering == "weight_asc":
            all_candidates.sort(key=lambda x: (x.bit_count(), x))
        elif ordering == "weight_desc":
            all_candidates.sort(key=lambda x: (-x.bit_count(), x))
        elif ordering == "value_asc":
            pass  # already in ascending order
        elif ordering == "value_desc":
            all_candidates.reverse()

        # Process candidates
        for cand in all_candidates:
            if len(words) >= TARGET:
                break
            if time.time() > DEADLINE:
                break
            if tried[cand]:
                continue

            # Check distance to all existing words
            # Use numpy for vectorized computation
            diffs = words_np ^ cand
            dmin = np.min([d.bit_count() for d in diffs]) if len(words_np) > 0 else N

            if dmin >= MIN_DIST:
                words.append(cand)
                words_np = np.append(words_np, cand)
                tried[cand] = True

    # Step 3: If we haven't reached target, try random candidates
    if len(words) < TARGET:
        import random
        random.seed(12345)
        words_np = np.array(words, dtype=np.uint32)

        while len(words) < TARGET:
            if time.time() > DEADLINE:
                break
            cand = random.randint(1, (1 << N) - 1)
            if tried[cand]:
                continue

            diffs = words_np ^ cand
            dmin = np.min([d.bit_count() for d in diffs]) if len(words_np) > 0 else N

            if dmin >= MIN_DIST:
                words.append(cand)
                words_np = np.append(words_np, cand)
                tried[cand] = True
            else:
                # If too many failures, break
                pass

    return words
# EVOLVE-BLOCK-END
