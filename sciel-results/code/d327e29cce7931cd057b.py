def construct():
    n = 26
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if i < 25 and j < 25:
                # Cyclic coloring for the first 25 vertices
                # Color based on difference mod 5
                c = (j - i) % 5
            else:
                # One of the vertices is 25
                # Let k be the other vertex
                k = i if j == 25 else j
                c = k % 5
            colors.append(c)
    return {'n': n, 'colors': colors}
