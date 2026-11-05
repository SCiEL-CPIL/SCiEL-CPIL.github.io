    E = np.sum(adj_np)
    def compute_cut(s):
        val = s @ (adj_np @ s)
        return 0.25 * (2.0 * E - val)
