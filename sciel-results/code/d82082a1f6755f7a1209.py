        while True:
            r1, r2, r3 = rng.choice(NP, 3, replace=False)
            if r1 != i and r2 != i and r3 != i:
                break
