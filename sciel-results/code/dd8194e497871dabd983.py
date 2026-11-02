<<<<<<< SEARCH
    def soft_covering_radius_and_grad(pts, grid, T_min=50.0, T_max=50.0):
        """
        Compute smooth approximation of covering radius and its gradient.
        Uses soft-min over points and soft-max over grid points.
        """
        M = len(grid)
        # Ensure points are normalized
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        # Distances: M x N
        dot = np.dot(grid, pts_norm.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)
        
        # Avoid division by zero in sin
        sin_dists = np.sin(ang_dists)
        sin_dists[sin_dists < 1e-10] = 1e-10

        # 1. Soft-min over points (axis 1) for each grid point
        # D_j = -1/T_min * log(sum_k exp(-T_min * d_jk))
        exp_neg_T_min_d = np.exp(-T_min * ang_dists)
        sum_exp_neg = np.sum(exp_neg_T_min_d, axis=1)
        D = -1.0 / T_min * np.log(sum_exp_neg)
        
        # Weights w_jk = exp(-T_min * d_jk) / sum_k exp(-T_min * d_jk)
        w_jk = exp_neg_T_min_d / sum_exp_neg[:, np.newaxis]

        # 2. Soft-max over grid points (axis 0)
        # R = 1/T_max * log(sum_j exp(T_max * D_j))
        exp_T_max_D = np.exp(T_max * D)
        sum_exp_pos = np.sum(exp_T_max_D)
        R = 1.0 / T_max * np.log(sum_exp_pos)
        
        # Weights W_j = exp(T_max * D_j) / sum_j exp(T_max * D_j)
        W_j = exp_T_max_D / sum_exp_pos

        # Gradient calculation
        # dR/dp_k = sum_j W_j * (dD_j/dp_k)
        # dD_j/dp_k = w_jk * (d d_jk / d p_k)
        # d d_jk / d p_k = proj_{p_k} (-grid_j / sin(d_jk))
        
        # Vectorized computation of gradient contributions
        # grad_contrib[j, k] = W_j * w_jk * (-grid_j / sin_dists[j, k])
        # Then project each row j,k to tangent space at p_k? 
        # No, the projection is specific to p_k.
        
        # Let's compute the raw gradient first:
        # raw_grad[j, k] = W_j * w_jk * (-grid_j / sin_dists[j, k])
        # This is an M x N x 3 tensor.
        
        # To save memory and time, we can sum over j first for each k.
        # G_k = sum_j W_j * w_jk * (-grid_j / sin_dists[j, k])
        
        # W_j is (M,), w_jk is (M, N), grid is (M, 3), sin_dists is (M, N).
        # Term for (j, k): W_j * w_jk * (-grid_j[:, np.newaxis] / sin_dists[:, np.newaxis])
        
        # We can compute this as:
        # weights_jk = W_j[:, np.newaxis] * w_jk  # M x N
        # inv_sin = 1.0 / sin_dists # M x N
        # grad_sum = np.sum(weights_jk[:, :, np.newaxis] * (-grid[:, np.newaxis, :] * inv_sin[:, :, np.newaxis]), axis=0)
        # This results in (N, 3)
        
        # Let's try a more efficient matmul approach if possible, or just use the loop if M is small enough.
        # M=8000, N=130. M*N = 1,040,000. This is manageable.
        
        weights_jk = W_j[:, np.newaxis] * w_jk
        # grad_raw[j, k] = weights_jk[j, k] * (-grid[j] / sin_dists[j, k])
        # We want G[k] = sum_j grad_raw[j, k]
        
        # grad_raw is (M, N, 3)
        # grad_raw = weights_jk[:, :, np.newaxis] * (-grid[:, np.newaxis, :] / sin_dists[:, :, np.newaxis])
        
        # This creates a large temporary array. Let's optimize.
        # G[k] = - sum_j (W_j * w_jk / sin_dists[j, k]) * grid[j]
        
        # Let A[j, k] = W_j * w_jk / sin_dists[j, k]
        # G[k] = - A[:, k] @ grid  (where A[:, k] is a vector of length M)
        
        A = weights_jk / sin_dists # M x N
        G = - (A.T @ grid) # N x 3
        
        # Now project G[k] to the tangent space at pts_norm[k]
        # proj_v(p) = v - (v . p) p
        dot_G_p = np.sum(G * pts_norm, axis=1, keepdims=True)
        G_tangent = G - dot_G_p * pts_norm
        
        return float(R), G_tangent.flatten()
=======
    def facet_objective_and_grad(pts, T=100.0):
        """
        Compute smooth approximation of max facet circumradius using log-sum-exp,
        and its gradient with respect to point positions.
        """
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
        except Exception:
            # Fallback: use a dense grid to find worst regions
            dot = np.dot(GRID_FINE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            # Soft max
            shifted = T * (min_dists - np.max(min_dists))
            exp_vals = np.exp(shifted)
            R = np.max(min_dists) + np.log(np.sum(exp_vals)) / T
            # Gradient: for each point, sum over grid points where it's the nearest
            # This is approximate
            G = np.zeros((len(pts), 3))
            for k in range(len(pts)):
                nearest_mask = np.argmin(ang_dists, axis=1) == k
                if np.any(nearest_mask):
                    grid_pts = GRID_FINE[nearest_mask]
                    dists = ang_dists[nearest_mask]
                    sin_dists = np.sin(dists)
                    sin_dists[sin_dists < 1e-10] = 1e-10
                    weights = exp_vals[nearest_mask] / np.sum(exp_vals)
                    G[k] = -np.sum(weights[:, np.newaxis] * grid_pts / sin_dists[:, np.newaxis], axis=0)
                    G[k] -= np.dot(G[k], pts[k]) * pts[k]
            return float(R), G.flatten()

        a = pts[faces[:, 0]]
        b = pts[faces[:, 1]]
        c = pts[faces[:, 2]]
        
        n_faces = len(faces)
        radii = np.zeros(n_faces)
        
        for i in range(n_faces):
            ai, bi, ci = a[i], b[i], c[i]
            ab = np.linalg.norm(ai - bi)
            bc = np.linalg.norm(bi - ci)
            ca = np.linalg.norm(ci - ai)
            area = 0.5 * np.linalg.norm(np.cross(ai - bi, ai - ci))
            if area < 1e-10:
                radii[i] = 0.5 * max(ab, bc, ca)
            else:
                radii[i] = (ab * bc * ca) / (4.0 * area)
        
        # Soft max of radii
        max_r = np.max(radii)
        shifted = T * (radii - max_r)
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        R = max_r + np.log(sum_exp) / T
        
        # Weights for each face
        W = exp_vals / sum_exp
        
        # Gradient: dR/dp_k = sum_i W_i * (dR_i/dp_k)
        # R_i = (ab_i * bc_i * ca_i) / (4 * area_i)
        # This is complex; use finite differences or analytic gradient
        
        # For simplicity, use a smoothed objective: maximize min distance to face circumcenters
        # Actually, let's just use the repulsion gradient as a proxy and rely on local search
        
        G = np.zeros((len(pts), 3))
        
        # Compute gradient analytically for each face
        for i in range(n_faces):
            if W[i] < 1e-10:
                continue
            idx = faces[i]
            ai, bi, ci = pts[idx[0]], pts[idx[1]], pts[idx[2]]
            
            ab = np.linalg.norm(ai - bi)
            bc = np.linalg.norm(bi - ci)
            ca = np.linalg.norm(ci - ai)
            area = 0.5 * np.linalg.norm(np.cross(ai - bi, ai - ci))
            
            if area < 1e-10:
                continue
            
            R_i = (ab * bc * ca) / (4.0 * area)
            
            # dR_i/dab = (bc*ca - ab*(bc*ca/(4*area) - 4*area*d(ab*bc*ca)/(4*area)))... complex
            # Use numerical gradient for this face
            eps = 1e-5
            dR_dai = np.zeros(3)
            dR_dbi = np.zeros(3)
            dR_dci = np.zeros(3)
            
            for d in range(3):
                # Perturb ai
                ai_p = ai.copy()
                ai_p[d] += eps
                ai_p -= np.dot(ai_p, ai) * ai
                ai_p /= np.linalg.norm(ai_p)
                
                ab_p = np.linalg.norm(ai_p - bi)
                bc_p = np.linalg.norm(bi - ci)
                ca_p = np.linalg.norm(ci - ai_p)
                area_p = 0.5 * np.linalg.norm(np.cross(ai_p - bi, ai_p - ci))
                if area_p > 1e-10:
                    R_p = (ab_p * bc_p * ca_p) / (4.0 * area_p)
                    dR_dai[d] = (R_p - R_i) / eps
                else:
                    dR_dai[d] = 0.0
                
                # Perturb bi
                bi_p = bi.copy()
                bi_p[d] += eps
                bi_p -= np.dot(bi_p, bi) * bi
                bi_p /= np.linalg.norm(bi_p)
                
                ab_p = np.linalg.norm(ai - bi_p)
                bc_p = np.linalg.norm(bi_p - ci)
                ca_p = np.linalg.norm(ci - ai)
                area_p = 0.5 * np.linalg.norm(np.cross(ai - bi_p, ai - ci))
                if area_p > 1e-10:
                    R_p = (ab_p * bc_p * ca_p) / (4.0 * area_p)
                    dR_dbi[d] = (R_p - R_i) / eps
                else:
                    dR_dbi[d] = 0.0
                
                # Perturb ci
                ci_p = ci.copy()
                ci_p[d] += eps
                ci_p -= np.dot(ci_p, ci) * ci
                ci_p /= np.linalg.norm(ci_p)
                
                ab_p = np.linalg.norm(ai - bi)
                bc_p = np.linalg.norm(bi - ci_p)
                ca_p = np.linalg.norm(ci_p - ai)
                area_p = 0.5 * np.linalg.norm(np.cross(ai - bi, ai - ci_p))
                if area_p > 1e-10:
                    R_p = (ab_p * bc_p * ca_p) / (4.0 * area_p)
                    dR_dci[d] = (R_p - R_i) / eps
                else:
                    dR_dci[d] = 0.0
            
            G[idx[0]] += W[i] * dR_dai
            G[idx[1]] += W[i] * dR_dbi
            G[idx[2]] += W[i] * dR_dci
        
        # Project to tangent space
        for k in range(len(pts)):
            G[k] -= np.dot(G[k], pts[k]) * pts[k]
        
        return float(R), G.flatten()
>>>>>>> REPLACE

