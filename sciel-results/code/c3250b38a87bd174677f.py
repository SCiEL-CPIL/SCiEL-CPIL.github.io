import numpy as np

def construct():
    N = 50
    # Initialize angles using Fibonacci sphere for uniform distribution
    theta = np.zeros(N)
    phi = np.zeros(N)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        # Map i to z-axis [-1, 1]
        z = 1.0 - 2.0 * (i + 0.5) / N
        theta[i] = np.arccos(z)
        phi[i] = i * golden_angle
    
    R = 0.35
    best_r = 0.0
    best_theta = theta.copy()
    best_phi = phi.copy()
    best_R = R

    max_iter = 20000
    step_size = 0.05
    delta_active = 0.02

    def get_positions(th, ph, rad):
        x = rad * np.sin(th) * np.cos(ph)
        y = rad * np.sin(th) * np.sin(ph)
        z = rad * np.cos(th)
        return np.column_stack((x, y, z))

    for i in range(max_iter):
        centers = get_positions(theta, phi, R)
        
        # Compute distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        abs_centers = np.abs(centers)
        max_coord = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_coord
        
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_r > best_r:
            best_r = current_r
            best_theta = theta.copy()
            best_phi = phi.copy()
            best_R = R
            
        # Identify active constraints
        active_pairs = np.argwhere(dist < 2 * (current_r + delta_active))
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        active_walls = np.where(wall_dists < current_r + delta_active)[0]
        
        # Gradients with respect to [R, theta_0...N-1, phi_0...N-1]
        # Total dimension: 1 + 2*N
        grad_R = 0.0
        grad_theta = np.zeros(N)
        grad_phi = np.zeros(N)
        
        # Pair contributions: Maximize distance
        # d/dR |ci - cj| = (ui - uj) . (ci - cj) / d
        # d/dtheta_i |ci - cj| = (dci/dtheta_i) . (ci - cj) / d
        # d/dphi_i |ci - cj| = (dci/dphi_i) . (ci - cj) / d
        
        for i_idx, j_idx in active_pairs:
            ci = centers[i_idx]
            cj = centers[j_idx]
            v = ci - cj
            d = np.linalg.norm(v)
            if d < 1e-9:
                continue
            v_unit = v / d
            
            # Derivatives of ci
            # dci/dR = ci / R
            # dci/dtheta = R * (cos(th)cos(ph), cos(th)sin(ph), -sin(th))
            # dci/dphi = R * (-sin(th)sin(ph), sin(th)cos(ph), 0)
            
            ui = ci / R
            dci_dR = ui
            dci_dth = R * np.array([np.cos(theta[i_idx])*np.cos(phi[i_idx]),
                                    np.cos(theta[i_idx])*np.sin(phi[i_idx]),
                                    -np.sin(theta[i_idx])])
            dci_dph = R * np.array([-np.sin(theta[i_idx])*np.sin(phi[i_idx]),
                                    np.sin(theta[i_idx])*np.cos(phi[i_idx]),
                                    0])
            
            uj = cj / R
            dcj_dR = uj
            dcj_dth = R * np.array([np.cos(theta[j_idx])*np.cos(phi[j_idx]),
                                    np.cos(theta[j_idx])*np.sin(phi[j_idx]),
                                    -np.sin(theta[j_idx])])
            dcj_dph = R * np.array([-np.sin(theta[j_idx])*np.sin(phi[j_idx]),
                                    np.sin(theta[j_idx])*np.cos(phi[j_idx]),
                                    0])
            
            # Gradient of distance
            # d(dist)/dR = dci_dR . v_unit - dcj_dR . v_unit
            # d(dist)/dtheta_i = dci_dth . v_unit
            # d(dist)/dtheta_j = -dcj_dth . v_unit  (since d(dist)/dcj = -v_unit)
            
            grad_R += np.dot(dci_dR, v_unit) - np.dot(dcj_dR, v_unit)
            grad_theta[i_idx] += np.dot(dci_dth, v_unit)
            grad_theta[j_idx] -= np.dot(dcj_dth, v_unit)
            
            grad_phi[i_idx] += np.dot(dci_dph, v_unit)
            grad_phi[j_idx] -= np.dot(dcj_dph, v_unit)

        # Wall contributions: Maximize wall_dist
        # w_i = 0.5 - sign(c_ik) * c_ik
        # dw/dR = -sign * dci_k/dR
        # dw/dtheta = -sign * dci_k/dtheta
        # dw/dphi = -sign * dci_k/dphi
        
        for i_idx in active_walls:
            ci = centers[i_idx]
            limiting_coords = np.argmax(np.abs(ci))
            sign_c = np.sign(ci[limiting_coords])
            
            # dci/dR
            dci_dR = ci / R
            dci_dth = R * np.array([np.cos(theta[i_idx])*np.cos(phi[i_idx]),
                                    np.cos(theta[i_idx])*np.sin(phi[i_idx]),
                                    -np.sin(theta[i_idx])])
            dci_dph = R * np.array([-np.sin(theta[i_idx])*np.sin(phi[i_idx]),
                                    np.sin(theta[i_idx])*np.cos(phi[i_idx]),
                                    0])
            
            # Extract k-th component
            dw_dR = -sign_c * dci_dR[limiting_coords]
            dw_dth = -sign_c * dci_dth[limiting_coords]
            dw_dph = -sign_c * dci_dph[limiting_coords]
            
            grad_R += dw_dR
            grad_theta[i_idx] += dw_dth
            grad_phi[i_idx] += dw_dph

        # Normalize gradient
        grad = np.concatenate(([grad_R], grad_theta, grad_phi))
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm > 1e-9:
            grad /= grad_norm
            R += step_size * grad[0]
            theta += step_size * grad[1:1+N]
            phi += step_size * grad[1+N:1+2*N]
            
            # Keep R positive and reasonable
            R = np.clip(R, 0.1, 0.49)
            
        # Decay
        step_size *= 0.999
        delta_active *= 0.999

    # Return the best centers found
    return get_positions(best_theta, best_phi, best_R)
