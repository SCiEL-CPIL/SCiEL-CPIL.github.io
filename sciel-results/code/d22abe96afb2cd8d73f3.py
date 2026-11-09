<<<<<<< SEARCH
    g = greedy()
    b = beam_search()

    candidates = [g, b]
    # also try a couple of randomized greedy orders for diversity
    import random
    rnd = random.Random(12345)
    for _ in range(3):
        order = list(range(24, 0, -1))
        rnd.shuffle(order)
        sums = {0}
        chosen = []
        for i in order:
            w = scale // i
            if w not in sums:
                chosen.append(i)
                sums = sums | {s + w for s in sums}
        candidates.append(sorted(chosen))
=======
    g = greedy()
    b = beam_search()

    candidates = [g, b]
    # try more randomized greedy orders for diversity
    import random
    rnd = random.Random(12345)
    for _ in range(10):
        order = list(range(24, 0, -1))
        rnd.shuffle(order)
        sums = {0}
        chosen = []
        for i in order:
            w = scale // i
            if w not in sums:
                chosen.append(i)
                sums = sums | {s + w for s in sums}
        candidates.append(sorted(chosen))
    
    # Also try greedy starting from smallest
    sums = {0}
    chosen_small = []
    for i in range(1, 25):
        w = scale // i
        if w not in sums:
            chosen_small.append(i)
            sums = sums | {s + w for s in sums}
    candidates.append(sorted(chosen_small))
>>>>>>> REPLACE

