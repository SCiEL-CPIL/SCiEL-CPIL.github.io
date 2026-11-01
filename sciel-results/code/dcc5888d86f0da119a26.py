<<<<<<< SEARCH
    # Local search: single swap
    def swap_search(blks, passes=40):
        if not blks:
            return blks, False
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
        sol_set = set()
        for b in blks:
            sol_set.add(tuple(b))
        improved = False
        for _ in range(passes):
            if time.time() > deadline - 5.0:
                break
            found = False
            for i in range(len(blks)):
                if time.time() > deadline - 5.0:
                    break
                ut = []
                for ts in itertools.combinations(blks[i], t):
                    idx = ts_index[ts]
                    if cc[idx] == 1:
                        ut.append(idx)
                if not ut:
                    continue
                cands = set(ts_to_block_idx[ut[0]])
                for ti in ut[1:]:
                    cands.intersection_update(ts_to_block_idx[ti])
                    if not cands:
                        break
                if not cands:
                    continue
                valid = [c for c in cands if tuple(all_k_blocks[c]) not in sol_set]
                if not valid:
                    continue
                ni = random.choice(valid)
                nb = list(all_k_blocks[ni])
                old = blks[i]
                for ts in itertools.combinations(old, t):
                    cc[ts_index[ts]] -= 1
                for ts in itertools.combinations(nb, t):
                    cc[ts_index[ts]] += 1
                sol_set.discard(tuple(old))
                sol_set.add(tuple(nb))
                blks[i] = nb
                improved = True
                found = True
                break
            if not found:
                break
        return blks, improved
=======
    # Precompute block tuples for fast lookup
    block_tuples = [tuple(b) for b in all_k_blocks]

    # Local search: single swap - optimized to find critical blocks first
    def swap_search(blks, passes=40):
        if not blks:
            return blks, False
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
        sol_set = set()
        for b in blks:
            sol_set.add(tuple(b))
        improved = False
        
        # Identify blocks with unique subsets (critical blocks)
        critical_blocks = []
        for i, b in enumerate(blks):
            has_unique = False
            for ts in itertools.combinations(b, t):
                if cc[ts_index[ts]] == 1:
                    has_unique = True
                    break
            if has_unique:
                critical_blocks.append(i)
        
        for _ in range(passes):
            if time.time() > deadline - 5.0:
                break
            found = False
            # Try critical blocks first, then others
            order = critical_blocks + [i for i in range(len(blks)) if i not in set(critical_blocks)]
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                if i >= len(blks):
                    continue
                ut = []
                for ts in itertools.combinations(blks[i], t):
                    idx = ts_index[ts]
                    if cc[idx] == 1:
                        ut.append(idx)
                if not ut:
                    continue
                cands = set(ts_to_block_idx[ut[0]])
                for ti in ut[1:]:
                    cands.intersection_update(ts_to_block_idx[ti])
                    if not cands:
                        break
                if not cands:
                    continue
                valid = [c for c in cands if block_tuples[c] not in sol_set]
                if not valid:
                    continue
                ni = random.choice(valid)
                nb = list(all_k_blocks[ni])
                old = blks[i]
                for ts in itertools.combinations(old, t):
                    cc[ts_index[ts]] -= 1
                for ts in itertools.combinations(nb, t):
                    cc[ts_index[ts]] += 1
                sol_set.discard(tuple(old))
                sol_set.add(tuple(nb))
                blks[i] = nb
                improved = True
                found = True
                break
            if not found:
                break
        return blks, improved
>>>>>>> REPLACE

