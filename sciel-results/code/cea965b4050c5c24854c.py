    # 8. Hexagonal lattice fragment
    h = np.sqrt(3) / 2
    hex_pts = []
    for r in range(4):
        for c in range(4):
            x = c + (r % 2) * 0.5
            y = r * h
            hex_pts.append([x, y])
    hex_arr = np.array(hex_pts)
    inits.append(hex_arr)
