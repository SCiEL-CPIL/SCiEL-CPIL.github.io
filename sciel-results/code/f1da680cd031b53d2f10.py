    def bron_kerbosch_pivot(adj, n):
        """Bron-Kerbosch with pivot to enumerate maximal cliques."""
        cliques = []

        def bk(R, P, X):
            if time.time() - start_time > DEADLINE:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return

            # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u

            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])

            for v in candidates:
                R.add(v)
                bk(R, P & adj[v], X & adj[v])
                R.discard(v)
                P.discard(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques
