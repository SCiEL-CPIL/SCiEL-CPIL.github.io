import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    rep = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    rep = (0, 1, (z * inv_y) % q)
                else:
                    rep = (0, 0, 1)
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)

    n = len(points)

    adj = [[False] * n for _ in range(n)]
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            val = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
            if val == 0:
                adj[i][j] = True
                adj[j][i] = True
                edges.append([i, j])

    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                neighbors[i].append(j)

    def count_c4s(adj, neighbors):
        count = 0
        for i in range(n):
            ni_set = set(neighbors[i])
            for j in range(i + 1, n):
                if not adj[i][j]:
                    common = 0
                    for w in neighbors[j]:
                        if w in ni_set:
                            common += 1
                    count += common // 2
        return count

    # Add best edge to reach 91 edges
    min_paths = float('inf')
    best_edge = None
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                count = 0
                for w1 in neighbors[i]:
                    for w2 in neighbors[j]:
                        if w1 != w2 and adj[w1][w2]:
                            count += 1
                if count < min_paths:
                    min_paths = count
                    best_edge = [i, j]

    if best_edge:
        edges.append(best_edge)
        adj[best_edge[0]][best_edge[1]] = True
        adj[best_edge[1]][best_edge[0]] = True
        neighbors[best_edge[0]].append(best_edge[1])
        neighbors[best_edge[1]].append(best_edge[0])

    # Local search: minimize C4s by swapping edges
    current_c4 = count_c4s(adj, neighbors)
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for ei in range(len(edges)):
            if time.time() > deadline - 30:
                break
            ei_u, ei_v = edges[ei]

            adj[ei_u][ei_v] = False
            adj[ei_v][ei_u] = False
            neighbors[ei_u].remove(ei_v)
            neighbors[ei_v].remove(ei_u)

            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i][j]:
                        adj[i][j] = True
                        adj[j][i] = True
                        neighbors[i].append(j)
                        neighbors[j].append(i)

                        new_c4 = count_c4s(adj, neighbors)
                        if new_c4 < current_c4:
                            current_c4 = new_c4
                            edges[ei] = [i, j]
                            improved = True

                        adj[i][j] = False
                        adj[j][i] = False
                        neighbors[i].remove(j)
                        neighbors[j].remove(i)

            adj[ei_u][ei_v] = True
            adj[ei_v][ei_u] = True
            neighbors[ei_u].append(ei_v)
            neighbors[ei_v].append(ei_u)

    return edges
