N = 40

def construct():
    # Two K20 cliques + perfect matching: 400 edges
    edge_set = set()
    for i in range(20):
        for j in range(i+1, 20):
            edge_set.add((i, j))
            edge_set.add((i+20, j+20))
        edge_set.add((i, i+20))
    edges = sorted(edge_set)

    def valid_cycle(cyc):
        if len(cyc) < 4 or cyc[0] != cyc[-1]:
            return False
        body = cyc[:-1]
        if len(set(body)) != len(body):
            return False
        for i in range(len(body)):
            u, v = body[i], body[(i+1) % len(body)]
            if (min(u,v), max(u,v)) not in edge_set:
                return False
        return True

    cycles = []
    for k in range(3, N+1):
        if k == 3:
            cyc = [0, 20, 1, 0]
        elif k == 4:
            cyc = [0, 20, 21, 1, 0]
        elif k <= 21:
            a = k - 2
            cyc = list(range(a)) + [20+a-1, 20, 0]
        else:
            a = k // 2
            b = k - a
            if b == a:
                path_a = list(range(a))
                path_b = list(range(20+a-1, 20-1, -1))
                cyc = path_a + path_b + [0]
            else:
                a2 = a + 1
                b2 = a
                path_a = list(range(a2))
                path_b = list(range(20+a2-1, 20+a2-b2-1, -1))
                cyc = path_a + path_b + [1]
        cycles.append(cyc)

    return {'edges': edges, 'cycles': cycles}
