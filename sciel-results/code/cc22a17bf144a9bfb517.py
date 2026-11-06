    random.seed(12345)
    np.random.seed(12345)
    
    # initial consideration of seed
    consider(set(range(64,129)))
    
    restart = 0
    while time.time() < deadline:
        A_init = greedy_cover()
        consider(A_init)
        # SA slice
        slice_end = min(time.time() + 6.0, deadline - 1.0)
        if time.time() < slice_end:
            local_A, local_peak = sa(A_init, slice_end)
            if local_A is not None:
                consider(local_A)
                # polish
                polished = polish(local_A)
                consider(polished)
        restart += 1
        # occasionally try different init sizes
        if restart % 5 == 0:
            # try a denser random init
            k = random.randint(15, 35)
            A_rand = set(random.sample(range(N), k))
            A_rand.add(128)
            # greedy cover from here
            A_init2 = greedy_cover_from(A_rand)  # need variant
            consider(A_init2)
    
    if best_A is None:
        return list(range(64,129))
    return sorted(best_A)
