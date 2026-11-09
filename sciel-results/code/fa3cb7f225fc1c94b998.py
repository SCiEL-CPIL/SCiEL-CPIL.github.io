                    valid_merge = True
                    for u in Li_set:
                        for v in Rj_set:
                            if v not in adj[u]:
                                valid_merge = False
                                break
                        if not valid_merge:
                            break
                    if valid_merge:
                        for u in Lj_set:
                            for v in Ri_set:
                                if v not in adj[u]:
                                    valid_merge = False
                                    break
                            if not valid_merge:
                                break
