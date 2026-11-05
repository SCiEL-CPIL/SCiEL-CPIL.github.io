def has_q3(adj):
    for idx_a in range(len(four_sets)):
        A = four_sets[idx_a]
        a_mask = four_masks[idx_a]
        for idx_b in range(len(four_sets)):
            B = four_sets[idx_b]
            b_mask = four_masks[idx_b]
            if a_mask & b_mask:
                continue
            # Count edges
            total = 0
            for a in A:
                total += bin(adj[a] & b_mask).count('1')
            if total != 12:
                continue
            # Check degrees
            ok = True
            for a in A:
                if bin(adj[a] & b_mask).count('1') != 3:
                    ok = False
                    break
            if ok:
                for b in B:
                    if bin(adj[b] & a_mask).count('1') != 3:
                        ok = False
                        break
            if not ok:
                continue
            # Perfect matching check
            mi, mj = [], []
            for i, a in enumerate(A):
                for j, b in enumerate(B):
                    if not (adj[a] & (1 << b)):
                        mi.append(i)
                        mj.append(j)
            if len(mi) == 4 and len(set(mi)) == 4 and len(set(mj)) == 4:
                return True
    return False
