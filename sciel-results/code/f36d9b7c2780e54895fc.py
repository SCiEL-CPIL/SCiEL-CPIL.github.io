    def get_neighbours(side, x_arr):
        y_coords = np.empty((q, d + 1), dtype=np.int64)
        y_coords[:, 0] = 0
        y_coords[:, 1] = free_vals
        if side == 0:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                y_coords[:, c] = (x_arr[c] + mul_table[y_coords[:, a], x_arr[b]]) % q
        else:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                y_coords[:, c] = (x_arr[c] - mul_table[x_arr[a], y_coords[:, b]]) % q
        return y_coords
