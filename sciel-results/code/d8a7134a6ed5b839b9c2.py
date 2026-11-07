def construct():
    # Fano plane: 7 edges on vertices 0-6, 3-chromatic, not 2-colorable
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
    ]
    # 3-coloring: v0=0, v1=1, v2=2, v3=0, v4=1, v5=2, v6=0, v7=0, v8=0
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    return {'edges': edges, 'colors': colors}
