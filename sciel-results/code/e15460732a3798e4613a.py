<<<<<<< SEARCH
    max_S = 0.0
    for i in range(len(inner_positions)):
        px, py, pa = inner_positions[i], inner_angles[i]
        verts = _hex_vertices(px, py, 1.0, pa)
=======
    max_S = 0.0
    for i in range(len(inner_positions)):
        px, py = inner_positions[i]
        pa = inner_angles[i]
        verts = _hex_vertices(px, py, 1.0, pa)
>>>>>>> REPLACE
