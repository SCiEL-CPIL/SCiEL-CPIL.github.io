            # Better greedy: Randomize order of all potential edges
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(pairs)
            for u, v in pairs:
                if not has_c4_conflict(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
