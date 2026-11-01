    def initial_grid():
        params = np.zeros(3*N)
        idx = 0
        # 3 rows
        for j in range(3):
            y = j * np.sqrt(3)
            # 4 columns
            for i in range(4):
                x = i * 2.0 + (j % 2) * 1.0
                params[3*idx : 3*idx+3] = [x, y, 0]
                idx += 1
        return params
