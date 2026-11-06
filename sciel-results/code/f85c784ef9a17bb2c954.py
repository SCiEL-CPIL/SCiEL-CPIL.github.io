    def polish_cover(cover, cliques, vertex_cliques):
        """Remove redundant vertices."""
        cover_set = set(cover)
        changed = True
        while changed:
            changed = False
            for v in list(cover_set):
                is_critical = False
                for c_idx in vertex_cliques[v]:
                    clique = cliques[c_idx]
                    # Check if any other vertex in clique is in cover_set
                    other_in_cover = False
                    for u in clique:
                        if u != v and u in cover_set:
                            other_in_cover = True
                            break
                    if not other_in_cover:
                        is_critical = True
                        break
                
                if not is_critical:
                    cover_set.remove(v)
                    changed = True
                    
        return cover_set
