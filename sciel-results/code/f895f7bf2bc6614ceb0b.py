        if restart == 0:
            # 4x5x5 grid
            ticks_x = (np.arange(4) + 0.5) / 4 - 0.5
            ticks_y = (np.arange(5) + 0.5) / 5 - 0.5
            ticks_z = (np.arange(5) + 0.5) / 5 - 0.5
            grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
            initial = grid.copy()
            initial += np.random.uniform(-0.01, 0.01, size=(N, 3))
        else:
            initial = np.random.uniform(-0.45, 0.45, size=(N, 3))
