                added = True
                while added:
                    added = False
                    for c in list(candidates):
                        clique.add(c)
                        candidates.remove(c)
                        new_candidates = set()
                        for x in candidates:
                            if x in rem_adj[c]:
                                new_candidates.add(x)
                        candidates = new_candidates
                        added = True
                        break
