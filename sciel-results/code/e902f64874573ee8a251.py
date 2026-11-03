    def consider(z):
        nonlocal best, best_val
        V = np.array([_hex_verts(z[3 * i], z[3 * i + 1], z[3 * i + 2]) for i in range(N)])
        if not _is_feasible(V):
            return
        v = _obj(z)
        if best_val is None or v < best_val:
            best_val = v
            best = z.copy()
    