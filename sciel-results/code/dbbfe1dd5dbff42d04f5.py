        res_colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                res_colors.append(int(colors[i, j]))
