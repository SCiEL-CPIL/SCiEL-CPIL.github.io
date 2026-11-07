    def check_diam_le_4(dist, u, v, n):
        for x in range(n):
            for y in range(n):
                d1 = dist[x][y]
                if d1 <= 4:
                    continue
                d2 = dist[x][u] + 1 + dist[v][y]
                d3 = dist[x][v] + 1 + dist[u][y]
                if min(d1, d2, d3) > 4:
                    return False
        return True
    