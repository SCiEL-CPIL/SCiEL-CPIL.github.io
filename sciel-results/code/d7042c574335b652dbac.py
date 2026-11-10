                if can_add:
                    clique.append(best_v)
                    candidates = [w for w in candidates if w != best_v and frozenset((best_v, w)) in rem]
                else:
                    candidates = [w for w in candidates if w != best_v]
