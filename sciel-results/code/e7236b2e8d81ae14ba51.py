<<<<<<< SEARCH
    def square_init():
        # 4x4 grid
        side = int(np.ceil(np.sqrt(N)))
        grid = np.array(np.meshgrid(range(side), range(side))).T.reshape(-1, 2)
        # Center it
        grid -= grid.mean(axis=0)
        return grid[:N]
=======
    def square_init():
        # 4x4 grid
        side = int(np.ceil(np.sqrt(N)))
        # Use float arrays to avoid integer casting errors during centering
        x = np.arange(side, dtype=float)
        y = np.arange(side, dtype=float)
        grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        # Center it
        grid -= grid.mean(axis=0)
        return grid[:N]
>>>>>>> REPLACE
