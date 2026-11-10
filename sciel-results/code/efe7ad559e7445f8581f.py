            # 1. Try to add
            added = False
            for z in range(n_nodes):
                if z in S_set: continue
                # check compatibility
                ok = True
                for u in S:
                    if conflict[z, u]:
                        ok = False
                        break
                if ok:
                    S.append(z)
                    S_set.add(z)
                    changed = True
                    added = True
                    break
    