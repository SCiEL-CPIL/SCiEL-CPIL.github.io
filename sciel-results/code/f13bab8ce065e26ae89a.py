        def count_c4s(adj):
            c4_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    common = np.sum(adj[i] * adj[j])
                    if common >= 2:
                        c4_count += common * (common - 1) // 2
            return c4_count // 2
        