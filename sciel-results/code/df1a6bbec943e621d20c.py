    def calc_side_length(positions):
        # positions: (N, 2)
        # Calculate max projection in each normal direction
        # proj: (3, N)
        proj = positions @ normals.T
        # max abs proj for each normal
        max_abs_proj = np.max(np.abs(proj), axis=1)
        # M is the max of these
        M = np.max(max_abs_proj)
        # Side length L
        L = (2.0 / np.sqrt(3.0)) * (M + np.sqrt(3.0)/2.0)
        return L
    