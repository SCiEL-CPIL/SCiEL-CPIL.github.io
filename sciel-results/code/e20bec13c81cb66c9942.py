<<<<<<< SEARCH
    # ---- Fixed local search: add missing moduli via residue swaps ----
    def local_search(selected, rng, time_budget=2.0):
        selected = list(selected)
        selected_moduli = set(m for _, m in selected)
        missing = [m for m in moduli if m not in selected_moduli]
        start = time.time()
        
        improved = True
        while improved and time.time() - start < time_budget and time.time() < deadline:
            improved = False
            rng.shuffle(missing)
            for target_m in missing:
                if time.time() - start >= time_budget or time.time() >= deadline:
                    break
                if target_m in selected_moduli:
                    continue
                # First: can we just add it?
                r_add = find_residue(target_m, selected, None)
                if r_add is not None:
                    selected.append((r_add, target_m))
                    selected_moduli.add(target_m)
                    improved = True
                    continue
                # Try single-swap: change residue of one existing class to free target_m
                added = False
                for i in range(len(selected)):
                    r_i, m_i = selected[i]
                    for r_i_new in range(m_i):
                        if r_i_new == r_i:
                            continue
                        # Check class i (with r_i_new) vs all others
                        ok_i = True
                        for j in range(len(selected)):
                            if j == i:
                                continue
                            r_j, m_j = selected[j]
                            g = gcd_of(m_i, m_j)
                            if r_i_new % g == r_j % g:
                                ok_i = False
                                break
                        if not ok_i:
                            continue
                        # Check target_m vs all (using r_i_new for class i's residue)
                        ok_t = True
                        for j in range(len(selected)):
                            r_j, m_j = selected[j]
                            g = gcd_of(target_m, m_j)
                            # For class i, use r_i_new; for others, use r_j
                            r_eff = r_i_new if j == i else r_j
                            if r_eff % g == r_j % g and j != i:
                                ok_t = False
                                break
                            if j == i:
                                # target vs class i: check target residue vs r_i_new mod gcd
                                # We need some residue r_t for target such that r_t % g != r_i_new % g
                                # This is possible iff not all residues 0..target_m-1 have same value mod g
                                # i.e., iff g > 1 (since if g==1, all residues are 0 mod 1, conflict always)
                                # Actually if g==1, gcd=1, so r_new%1==0==r_k%1 always -> incompatible
                                # So if g==1, we can NEVER add target_m with any residue
                                if g == 1:
                                    ok_t = False
                                    break
                                # If g>1, there exists r_t in [0,target_m) with r_t%g != r_i_new%g
                                # as long as target_m > g (which it is since target_m >= 2 and g <= target_m)
                                # Actually we need to verify: is there an r_t in [0, target_m-1] with r_t % g != r_i_new % g?
                                # Yes, since target_m >= 2 and g >= 2 (if g==1 we already handled), 
                                # the residues 0,1,...,target_m-1 cover multiple values mod g if target_m > g,
                                # or if target_m <= g, they cover target_m distinct values mod g, and since g >= 2,
                                # at most one value equals r_i_new % g, so there's another.
                                # So ok_t stays True for this j.
                                pass
                        if ok_t:
                            # Find a valid residue for target_m
                            r_t = None
                            for rt in range(target_m):
                                compatible = True
                                for j in range(len(selected)):
                                    r_j, m_j = selected[j]
                                    g = gcd_of(target_m, m_j)
                                    r_eff = r_i_new if j == i else r_j
                                    if rt % g == r_eff % g:
                                        compatible = False
                                        break
                                if compatible:
                                    r_t = rt
                                    break
                            if r_t is not None:
                                selected[i] = (r_i_new, m_i)
                                selected.append((r_t, target_m))
                                selected_moduli.add(target_m)
                                added = True
                                improved = True
                                break
                    if added:
                        break
                if time.time() - start >= time_budget or time.time() >= deadline:
                    break
        return selected
=======
    # ---- Optimized local search: add missing moduli via residue swaps ----
    def local_search(selected, rng, time_budget=2.0):
        selected = list(selected)
        selected_moduli = set(m for _, m in selected)
        missing = [m for m in moduli if m not in selected_moduli]
        start = time.time()
        
        improved = True
        while improved and time.time() - start < time_budget and time.time() < deadline:
            improved = False
            rng.shuffle(missing)
            for target_m in missing:
                if time.time() - start >= time_budget or time.time() >= deadline:
                    break
                if target_m in selected_moduli:
                    continue
                # First: can we just add it?
                r_add = find_residue(target_m, selected, None)
                if r_add is not None:
                    selected.append((r_add, target_m))
                    selected_moduli.add(target_m)
                    improved = True
                    continue
                # Try single-swap: change residue of one existing class to free target_m
                added = False
                for i in range(len(selected)):
                    r_i, m_i = selected[i]
                    for r_i_new in range(m_i):
                        if r_i_new == r_i:
                            continue
                        # Check class i (with r_i_new) vs all others
                        ok_i = True
                        for j in range(len(selected)):
                            if j == i:
                                continue
                            r_j, m_j = selected[j]
                            g = gcd_of(m_i, m_j)
                            if r_i_new % g == r_j % g:
                                ok_i = False
                                break
                        if not ok_i:
                            continue
                        # Find valid residue for target_m using blocked array
                        blocked = [False] * target_m
                        impossible = False
                        for j in range(len(selected)):
                            r_j, m_j = selected[j]
                            g = gcd_of(target_m, m_j)
                            if g == 1:
                                impossible = True
                                break
                            r_eff = r_i_new if j == i else r_j
                            rem = r_eff % g
                            for rt in range(rem, target_m, g):
                                blocked[rt] = True
                        if impossible:
                            continue
                        # Find an unblocked residue
                        r_t = None
                        for rt in range(target_m):
                            if not blocked[rt]:
                                r_t = rt
                                break
                        if r_t is not None:
                            selected[i] = (r_i_new, m_i)
                            selected.append((r_t, target_m))
                            selected_moduli.add(target_m)
                            added = True
                            improved = True
                            break
                    if added:
                        break
                if time.time() - start >= time_budget or time.time() >= deadline:
                    break
        return selected
>>>>>>> REPLACE

