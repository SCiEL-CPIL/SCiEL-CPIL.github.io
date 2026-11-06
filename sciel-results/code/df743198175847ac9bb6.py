            for v in list(candidates_for_recursion):
                bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
                P_set.remove(v)
                X.add(v)
