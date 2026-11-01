                candidates = set(subset_blocks[start_s])
                for s in critical_subsets:
                    if s == start_s: continue
                    for b in list(candidates):
                        if b not in subset_blocks[s]:
                            candidates.discard(b)
