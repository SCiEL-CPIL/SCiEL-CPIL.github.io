def construct():
        n = 15
        edges = []
        # Star centered at 0
        for i in range(1, n):
            edges.append([0, i])
        # Add one edge between leaves 1 and 2
        edges.append([1, 2])
        return edges
