    def separation_sat(pos1, th1, pos2, th2):
        d = pos2 - pos1
        max_overlap = 0.0
        # Normals of hex1 (edge normals)
        for k in range(6):
            a = th1 + np.pi/6 + k * np.pi / 3.0
            n = np.array([np.cos(a), np.sin(a)])
            h1_n = hex_support(th1, n)
            h2_mn = hex_support(th2, -n)
            d_proj = d @ n
            overlap = h1_n + h2_mn - d_proj
            if overlap > max_overlap:
                max_overlap = overlap
        
        # Normals of hex2 (edge normals)
        for k in range(6):
            a = th2 + np.pi/6 + k * np.pi / 3.0
            n = np.array([np.cos(a), np.sin(a)])
            h1_n = hex_support(th1, n)
            h2_mn = hex_support(th2, -n)
            d_proj = d @ n
            overlap = h1_n + h2_mn - d_proj
            if overlap > max_overlap:
                max_overlap = overlap
                
        return max_overlap
