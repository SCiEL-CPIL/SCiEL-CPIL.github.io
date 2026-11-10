# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses quaternion-based SO(3) parameterization for more effective optimization.
    """
    start_time = time.time()
    deadline = 570  # 600s limit with 30s safety margin
    
    n = 14
    d = 3
    
    def quat_to_rot(q):
        """Convert unit quaternion to 3x3 rotation matrix."""
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        return R
    
    def quat_mult(q1, q2):
        """Hamiltonian product of two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quat_conj(q):
        """Conjugate of quaternion."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def quat_norm(q):
        return np.linalg.norm(q)
    
    def quat_normalize(q):
        n = quat_norm(q)
        if n < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / n
    
    def points_from_quats(quats, base_points):
        """Apply rotations from quaternions to base points."""
        points = np.zeros((n, d))
        for i in range(n):
            R = quat_to_rot(quats[i])
            points[i] = R @ base_points[i]
        return points
    
    def compute_score(points):
        """Compute the metric (d_min/d_max)^2."""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2
    
    def gradient_ascent_quat(quats, base_points, lr=0.01, max_iter=200, tol=1e-8):
        """
        Gradient ascent on minimum pairwise distance using quaternion representation.
        Returns improved quaternions.
        """
        quats = quats.copy()
        best_score = compute_score(points_from_quats(quats, base_points))
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Current points and distances
            points = points_from_quats(quats, base_points)
            dists = pdist(points)
            if len(dists) == 0:
                break
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-10:
                break
            current_score = (d_min / d_max) ** 2
            
            # Find the minimum distance pair
            # pdist returns condensed form: distance between point i and j (i<j) is at index
            # idx = i*n - i*(i+1)//2 + (j-i-1)
            min_idx = np.argmin(dists)
            # Invert condensed index to get (i, j)
            # For n points, condensed index k corresponds to:
            # i = floor((n-1) - sqrt(2*k - 2*n + 1 - 2*(n-1)))  ... complicated
            # Simpler: build mapping
            i, j = 0, 1
            count = 0
            for ii in range(n):
                for jj in range(ii+1, n):
                    if count == min_idx:
                        i, j = ii, jj
                        break
                    count += 1
                if i == ii and j == jj:
                    break
            
            # Gradient of d_min with respect to points i and j
            # d/dp_i ||p_i - p_j|| = (p_i - p_j) / ||p_i - p_j||
            diff = points[i] - points[j]
            dist_ij = np.linalg.norm(diff)
            if dist_ij < 1e-10:
                break
            grad_i = diff / dist_ij
            grad_j = -diff / dist_ij
            
            # For other points, gradient is 0 (they don't affect d_min directly)
            # But we should also consider the effect on d_max for the ratio
            # For now, focus on maximizing d_min
            
            # Compute gradient with respect to quaternions
            # d(p_k)/dq_k = R'(q_k) @ base_points[k]
            # where R'(q) is the derivative of rotation matrix w.r.t. quaternion
            
            # Update quaternions for points i and j
            for idx, grad in [(i, grad_i), (j, grad_j)]:
                # Compute derivative of R @ base_point w.r.t. quaternion
                # R(q) = [[1-2(y^2+z^2), 2(xy-wz), 2(xz+wy)],
                #          [2(xy+wz), 1-2(x^2+z^2), 2(yz-wx)],
                #          [2(xz-wy), 2(yz+wx), 1-2(x^2+y^2)]]
                
                q = quats[idx]
                w, x, y, z = q
                bp = base_points[idx]
                
                # Gradient of each component of R @ bp w.r.t. (w, x, y, z)
                # R @ bp = [R[0,0]*bp[0] + R[0,1]*bp[1] + R[0,2]*bp[2], ...]
                
                # d(R[0,0])/dw = 0, d(R[0,0])/dx = 0, d(R[0,0])/dy = -4y, d(R[0,0])/dz = -4z
                # d(R[0,1])/dw = -2z, d(R[0,1])/dx = 2y, d(R[0,1])/dy = 2x, d(R[0,1])/dz = -2w
                # d(R[0,2])/dw = 2y, d(R[0,2])/dx = 2z, d(R[0,2])/dy = 2w, d(R[0,2])/dz = 2x
                
                # d(R[1,0])/dw = 2z, d(R[1,0])/dx = 2y, d(R[1,0])/dy = 2x, d(R[1,0])/dz = 2w
                # d(R[1,1])/dw = 0, d(R[1,1])/dx = -4x, d(R[1,1])/dy = 0, d(R[1,1])/dz = -4z
                # d(R[1,2])/dw = -2x, d(R[1,2])/dx = -2w, d(R[1,2])/dy = 2z, d(R[1,2])/dz = 2y
                
                # d(R[2,0])/dw = -2y, d(R[2,0])/dx = 2z, d(R[2,0])/dy = 2w, d(R[2,0])/dz = 2x
                # d(R[2,1])/dw = 2x, d(R[2,1])/dx = 2w, d(R[2,1])/dy = 2z, d(R[2,1])/dz = 2y
                # d(R[2,2])/dw = 0, d(R[2,2])/dx = -4x, d(R[2,2])/dy = -4y, d(R[2,2])/dz = 0
                
                # Gradient w.r.t. quaternion components
                grad_w = 0.0
                grad_x = 0.0
                grad_y = 0.0
                grad_z = 0.0
                
                # Contribution from bp[0]
                grad_w += 0.0 * bp[0] * grad[0] + 2.0 * z * bp[0] * grad[1] - 2.0 * y * bp[0] * grad[2]
                grad_x += 0.0 * bp[0] * grad[0] + 2.0 * y * bp[0] * grad[1] - 2.0 * w * bp[0] * grad[2]
                grad_y += -4.0 * y * bp[0] * grad[0] + 2.0 * x * bp[0] * grad[1] + 2.0 * w * bp[0] * grad[2]
                grad_z += -4.0 * z * bp[0] * grad[0] - 2.0 * w * bp[0] * grad[1] + 2.0 * x * bp[0] * grad[2]
                
                # Contribution from bp[1]
                grad_w += -2.0 * z * bp[1] * grad[0] + 0.0 * bp[1] * grad[1] + 2.0 * x * bp[1] * grad[2]
                grad_x += 2.0 * y * bp[1] * grad[0] - 4.0 * x * bp[1] * grad[1] - 2.0 * w * bp[1] * grad[2]
                grad_y += 2.0 * x * bp[1] * grad[0] + 0.0 * bp[1] * grad[1] + 2.0 * z * bp[1] * grad[2]
                grad_z += -2.0 * w * bp[1] * grad[0] - 2.0 * x * bp[1] * grad[1] + 2.0 * y * bp[1] * grad[2]
                
                # Contribution from bp[2]
                grad_w += 2.0 * y * bp[2] * grad[0] - 2.0 * x * bp[2] * grad[1] + 0.0 * bp[2] * grad[2]
                grad_x += 2.0 * z * bp[2] * grad[0] - 2.0 * w * bp[2] * grad[1] - 4.0 * x * bp[2] * grad[2]
                grad_y += 2.0 * w * bp[2] * grad[0] + 2.0 * z * bp[2] * grad[1] - 4.0 * y * bp[2] * grad[2]
                grad_z += 2.0 * x * bp[2] * grad[0] + 2.0 * y * bp[2] * grad[1] + 0.0 * bp[2] * grad[2]
                
                # Update quaternion
                quats[idx] += lr * np.array([grad_w, grad_x, grad_y, grad_z])
                quats[idx] = quat_normalize(quats[idx])
            
            # Check convergence
            new_points = points_from_quats(quats, base_points)
            new_dists = pdist(new_points)
            new_d_min = np.min(new_dists)
            new_d_max = np.max(new_dists)
            new_score = (new_d_min / new_d_max) ** 2 if new_d_max > 1e-10 else 0.0
            
            if new_score > best_score:
                best_score = new_score
            else:
                # Revert if score decreased
                quats = quats - lr * np.array([grad_w, grad_x, grad_y, grad_z])
                quats = quat_normalize(quats)
            
            if abs(new_score - best_score) < tol:
                break
        
        return quats
    
    # Initialize with icosahedron + 2 opposite points
    phi = (1 + np.sqrt(5)) / 2
    ico_coords = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_norms = np.linalg.norm(ico_coords, axis=1)
    ico_coords = ico_coords / ico_norms[:, np.newaxis]
    
    v1 = np.array([1, 1, 1])
    v1 = v1 / np.linalg.norm(v1)
    v2 = -v1
    
    base_points = np.vstack([ico_coords, [v1], [v2]])
    
    # Initialize quaternions to identity
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    quats = np.tile(identity_quat, (n, 1))
    
    best_score = -np.inf
    best_points = base_points
    
    # Run gradient ascent with different learning rates
    for lr in [0.05, 0.01, 0.005, 0.001]:
        if time.time() > deadline - 10:
            break
        
        quats_copy = quats.copy()
        quats_copy = gradient_ascent_quat(quats_copy, base_points, lr=lr, max_iter=100)
        
        points = points_from_quats(quats_copy, base_points)
        score = compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points
            quats = quats_copy
    
    # If we have time, do random restarts with perturbed quaternions
    while time.time() < deadline - 20:
        np.random.seed(int(time.time() * 1000) % 100000)
        
        # Random small perturbations to quaternions
        quats_pert = quats.copy()
        for i in range(n):
            pert = np.random.randn(4) * 0.1
            quats_pert[i] = quat_mult(quats_pert[i], quat_normalize(np.array([1.0, pert[1], pert[2], pert[3]])))
            quats_pert[i] = quat_normalize(quats_pert[i])
        
        quats_pert = gradient_ascent_quat(quats_pert, base_points, lr=0.01, max_iter=50)
        
        points = points_from_quats(quats_pert, base_points)
        score = compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points
            quats = quats_pert
    
    # Final check
    if not np.all(np.isfinite(best_points)):
        return np.zeros((14, 3))
    
    return best_points

# EVOLVE-BLOCK-END
