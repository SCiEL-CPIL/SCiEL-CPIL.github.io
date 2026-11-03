<<<<<<< SEARCH
    def calc_S_for_beta(pts, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = pts @ normals.T
        D = float(np.max(projs) - np.min(projs)) / 2.0
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D
=======
    def calc_S_for_beta(pts, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_proj = float(np.max(np.abs(projs)))
        S = (max_proj + APOTHEM) / APOTHEM
        return S, (cx, cy), max_proj
>>>>>>> REPLACE

