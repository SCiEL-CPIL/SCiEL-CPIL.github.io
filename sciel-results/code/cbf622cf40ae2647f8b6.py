    # independent pixel list (upper triangle of 16x16)
    indep = [(i,j) for i in range(16) for j in range(i,16)]
    R = len(indep)  # 138
    
    seeds = [
        make_seed_Q('disk',0.20),
        make_seed_Q('disk',0.23),
        make_seed_Q('disk',0.26),
        make_seed_Q('ring',0.28,0.12),
        make_seed_Q('square',0.22),
        make_seed_Q('super',0.24),
        make_seed_Q('random',0),
        make_seed_Q('random',0),
    ]
    best_Q = None; best_score = -1e9
    for s in seeds:
        cell = build_cell(s)
        sc = surrogate_ratio(cell)
        if sc > best_score:
            best_score = sc; best_Q = s.copy()
    
    current = best_Q.copy()
    cur_score = best_score
    no_improve = 0
    R_cand = 6
    while time.time() < DEADLINE:
        # best-of-R_cand random flip
        best_cand = None; best_cand_score = cur_score
        for _ in range(R_cand):
            p = np.random.randint(R)
            i,j = indep[p]
            trial = current.copy()
            trial[i,j] = 1 - trial[i,j]
            trial[j,i] = trial[i,j]
            cell = build_cell(trial)
            sc = surrogate_ratio(cell)
            if sc > best_cand_score:
                best_cand_score = sc; best_cand = trial
        if best_cand is not None:
            current = best_cand; cur_score = best_cand_score
            no_improve = 0
            if cur_score > best_score:
                best_score = cur_score; best_Q = current.copy()
        else:
            no_improve += 1
        if no_improve >= 12:
            # perturb
            npix = np.random.randint(3, 9)
            current = best_Q.copy()
            for _ in range(npix):
                p = np.random.randint(R)
                i,j = indep[p]
                current[i,j] = 1-current[i,j]
                current[j,i] = current[i,j]
            no_improve = 0
    cell = build_cell(best_Q)
    return cell.tolist()
