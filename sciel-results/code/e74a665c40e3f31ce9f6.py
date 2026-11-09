    def expand_clique(start_vertices, adj_set, forbidden, candidate_pool):
        clique = list(start_vertices)
        # Filter candidate_pool to those connected to all in clique
        candidates = [v for v in candidate_pool if v not in clique and all((min(u,v), max(u,v)) in adj_set and (min(u,v), max(u,v)) not in forbidden for u in clique)]
        while candidates:
            best_c = None
            best_score = -1
            for u in candidates:
                nb_count = 0
                for v in candidates:
                    if u == v: continue
                    if (min(u, v), max(u, v)) in adj_set and (min(u, v), max(u, v)) not in forbidden:
                        nb_count += 1
                if nb_count > best_score:
                    best_score = nb_count
                    best_c = u
            if best_c is None: break
            clique.append(best_c)
            new_candidates = []
            for u in candidates:
                if u == best_c: continue
                if (min(u, best_c), max(u, best_c)) in adj_set and (min(u, best_c), max(u, best_c)) not in forbidden:
                    new_candidates.append(u)
            candidates = new_candidates
        return clique
    