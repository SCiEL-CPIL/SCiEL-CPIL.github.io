def construct():
    # Disjoint union of 8 cliques covering 35 vertices.
    # No cycles at all => no C5. Independence number = number of cliques = 8 < 9.
    # Clique sizes: three of size 5, five of size 4. Total = 3*5 + 5*4 = 35.
    sizes = [5, 5, 5, 4, 4, 4, 4, 4]
    n = sum(sizes)
    edges = []
    offset = 0
    for s in sizes:
        for i in range(s):
            for j in range(i + 1, s):
                edges.append([offset + i, offset + j])
        offset += s
    return {'n': n, 'edges': edges}
