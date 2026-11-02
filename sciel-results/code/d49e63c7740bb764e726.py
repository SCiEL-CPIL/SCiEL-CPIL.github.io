<<<<<<< SEARCH
    else:
        # Fallback: sorted by distance
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]
=======
    elif layout_idx == 3:
        # Alternative layout: 2-4-5?
        # Row 0 (y=-1.5): 2 hexes
        # Row 1 (y=0): 4 hexes
        # Row 2 (y=1.5): 5 hexes
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 1):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k*_SQRT3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 3):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        # Fallback: sorted by distance
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]
>>>>>>> REPLACE
