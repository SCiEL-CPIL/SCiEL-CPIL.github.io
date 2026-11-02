    def calc_S_for_beta(pts, beta):
        """Vectorized LP for minimum enclosing hexagon with given rotation."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        n_pts = len(pts)
        # Construct A_ub and b_ub
        # 12 * n_pts constraints
        A_ub = np.zeros((12 * n_pts, 3))
        b_ub = np.zeros(12 * n_pts)

        projs = pts @ normals.T # (n_pts, 6)
        
        # We need to fill A_ub and b_ub
        # For k in 0..5:
        #   Rows 2*k*n_pts to 2*k*n_pts+n_pts: -n_k, -1, b = -projs[:, k]
        #   Rows 2*k*n_pts+n_pts to 2*k*n_pts+2*n_pts: n_k, -1, b = projs[:, k]
        
        # Vectorized fill
        # normals is (6, 2)
        # We want to repeat each normal n_pts times, but in blocks?
        # No, the structure is:
        # Block k (size 2*n_pts):
        #   First n_pts rows: -normal[k]
        #   Next n_pts rows: normal[k]
        
        # Let's create a tiled normals array
        # normals_tiled = np.tile(normals, (n_pts, 1)) # (n_pts, 6, 2)
        # This doesn't map directly to the row order.
        
        # The row order is:
        # k=0: n_pts rows of -n0, n_pts rows of n0
        # k=1: n_pts rows of -n1, n_pts rows of n1
        # ...
        
        # So we can create:
        # neg_normals = -normals # (6, 2)
        # pos_normals = normals # (6, 2)
        
        # We need to interleave them in blocks of n_pts.
        # This is hard to vectorize with simple tile/repeat without reshaping tricks.
        # Given n_pts=11 and 6 normals, the loop is only 6 iterations.
        # The loop overhead is negligible compared to LP solve.
        # I will keep the loop for clarity and safety, but ensure it's tight.
        
        for k in range(6):
            nx, ny = normals[k]
            projs_k = projs[:, k]
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs_k
            
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs_k

        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None
