def construct():
        n = 16
        # Partitions 0..7 and 8..15
        A = list(range(0, 8))
        B = list(range(8, 16))
        edges = []
        for a in A:
            for b in B:
                edges.append([a, b])
        return edges
