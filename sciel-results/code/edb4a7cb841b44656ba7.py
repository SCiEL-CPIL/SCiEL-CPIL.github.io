import math
import time
import random
import numpy as np

N = 12

def construct():
    deadline = time.time() + 145.0
    random.seed(42)
    
    def hex_support(theta, angle_deg):
        """Support function of unit hexagon in direction theta.
        Returns max projection of hexagon vertices in direction theta."""
        angle = math.radians(angle_deg)
        max_proj = 0.0
        for i in range(6):
            vert_angle = angle + i * math.pi / 3
            proj = math.cos(vert_angle - theta)
            if proj > max_proj:
                max_proj = proj
        return max_proj
    
    def outer_side_fast(centers, angles, ocx, ocy, otheta_deg):
        """Compute outer hexagon side length using support functions.
        For a regular hexagon with side s, the width in direction theta is s*sqrt(3)*|cos(theta - otheta)|.
        Actually, for a regular hexagon with vertices at angles otheta + k*pi/3,
        the support in direction phi is s*cos(min_k |phi - (otheta + k*pi/3)|) where we take the max.
        
        Actually the support function of a regular hexagon with circumradius R (= side) 
        in direction phi is R * cos(delta) where delta is the angle to the nearest vertex direction...
        No wait. Let me think more carefully.
        
        A regular hexagon with side length s has circumradius s.
        The vertices are at angles otheta + k*pi/3 for k=0..5, at distance s from center.
        The support function h(phi) = max over vertices of (vertex - center) · n_hat(phi)
        = s * max_k cos(phi - otheta - k*pi/3)
        = s * cos(min_k |phi - otheta - k*pi/3|)  [since cos is even and periodic]
        
        But actually the support function of a convex polygon is the max projection.
        For a regular hexagon, h(phi) = s * cos(delta) where delta is the angular distance 
        from phi to the nearest vertex direction, but capped at pi/6.
        
        Actually: h(phi) = s * max_k cos(phi - otheta - k*pi/3)
        Since the vertices are spaced pi/3 apart, the max of cos over these 6 directions
        is cos(min_k |phi - otheta - k*pi/3|) where we reduce to [0, pi/6].
        
        The outer hexagon contains all inner hexagons iff for all 6 face normals of the outer hexagon,
        the support of the union of inner hexagons in that direction <= support of outer hexagon.
        
        The face normals of the outer hexagon are at angles otheta + pi/6 + k*pi/3 for k=0..5.
        
        For the outer hexagon with side s and center (ocx, ocy), the support in direction n is:
        s * h_outer(n) where h_outer is the support of a unit hexagon (side 1) centered at origin.
        
        Wait, I need to be more careful. Let me use the definition:
        
        The outer hexagon has vertices at:
        v_k = (ocx, ocy) + s * (cos(otheta + k*pi/3), sin(otheta + k*pi/3)) for k=0..5
        
        The support function of this hexagon in direction n is:
        h_outer(n) = max_k (v_k - (ocx,ocy)) · n + (ocx,ocy) · n
                   = s * max_k cos(otheta + k*pi/3 - phi_n) + (ocx,ocy) · n
        
        where phi_n is the angle of direction n.
        
        For the union of inner hexagons, the support in direction n is:
        h_inner(n) = max_i [ (center_i) · n + s_i * max_k cos(angle_i + k*pi/3 - phi_n) ]
        where s_i = 1 for all inner hexagons.
        
        The outer hexagon contains the union iff h_outer(n) >= h_inner(n) for all n.
        By convexity, it suffices to check the 6 face normals of the outer hexagon.
        
        The face normals are perpendicular to the edges. The edges are between consecutive vertices.
        The edge from v_k to v_{k+1} has direction otheta + k*pi/3 + pi/6 (midway).
        The outward normal is at angle otheta + k*pi/3 + pi/6.
        
        So for each k=0..5, let n_k = otheta + k*pi/3 + pi/6.
        We need: h_outer(n_k) >= h_inner(n_k) for all k.
        
        h_outer(n_k) = s * max_j cos(otheta + j*pi/3 - n_k) + (ocx,ocy) · n_hat(n_k)
        Since n_k = otheta + k*pi/3 + pi/6, we have:
        otheta + j*pi/3 - n_k = (j-k)*pi/3 - pi/6
        max_j cos((j-k)*pi/3 - pi/6) = cos(pi/6) = sqrt(3)/2 (achieved at j=k and j=k+1... wait)
        
        Let me compute: for j=k: cos(-pi/6) = cos(pi/6) = sqrt(3)/2
        for j=k+1: cos(pi/3 - pi/6) = cos(pi/6) = sqrt(3)/2
        for j=k-1: cos(-pi/3 - pi/6) = cos(-pi/2) = 0
        for j=k+2: cos(2*pi/3 - pi/6) = cos(pi/2) = 0
        
        So max_j cos(...) = sqrt(3)/2.
        
        Therefore h_outer(n_k) = s * sqrt(3)/2 + (ocx*cos(n_k) + ocy*sin(n_k))
        
        And h_inner(n_k) = max_i [ cx_i*cos(n_k) + cy_i*sin(n_k) + hex_support(n_k, angle_i) ]
        
        So the constraint is:
        s * sqrt(3)/2 + ocx*cos(n_k) + ocy*sin(n_k) >= max_i [ cx_i*cos(n_k) + cy_i*sin(n_k) + hex_support(n_k, angle_i) ]
        
        Rearranging:
        s * sqrt(3)/2 >= max_i [ (cx_i - ocx)*cos(n_k) + (cy_i - ocy)*sin(n_k) + hex_support(n_k, angle_i) ]
        
        So: s >= (2/sqrt(3)) * max_k max_i [ (cx_i - ocx)*cos(n_k) + (cy_i - ocy)*sin(n_k) + hex_support(n_k, angle_i) ]
        
        This is O(6*12*6) = O(432) per evaluation, which is fast!
        """
        otheta = math.radians(otheta_deg)
        max_val = 0.0
        for k in range(6):
            nk = otheta + k * math.pi / 3 + math.pi / 6
            cnk = math.cos(nk)
            snk = math.sin(nk)
            for i in range(len(centers)):
                cx, cy = centers[i]
                a = angles[i]
                proj = (cx - ocx) * cnk + (cy - ocy) * snk + hex_support(nk, a)
                if proj > max_val:
                    max_val = proj
        return 2.0 * max_val / math.sqrt(3)
    
    def check_disjoint(centers, angles, min_dist=math.sqrt(3) - 0.001):
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist*min_dist:
                    return False
        return True
    
    def optimize(centers, angles, max_time=140.0):
        """Local optimization using gradient-free methods."""
        start_time = time.time()
        best_side = outer_side_fast(centers, angles, 0, 0, 0)
        best_centers = [list(c) for c in centers]
        best_angles = list(angles)
        best_ocx, best_ocy, best_otheta = 0.0, 0.0, 0.0
        
        # First, optimize outer hexagon parameters
        for outer_iter in range(200):
            if time.time() > start_time + max_time - 30:
                break
            improved = False
            # Try small perturbations of outer parameters
            for dx in [-0.1, 0, 0.1]:
                for dy in [-0.1, 0, 0.1]:
                    for da in [-5, 0, 5]:
                        side = outer_side_fast(best_centers, best_angles, best_ocx+dx, best_ocy+dy, best_otheta+da)
                        if side < best_side - 1e-10:
                            best_side = side
                            best_ocx += dx
                            best_ocy += dy
                            best_otheta += da
                            improved = True
            if not improved:
                break
        
        # Now optimize inner positions and angles
        step = 0.05
        for it in range(500):
            if time.time() > start_time + max_time - 10:
                break
            improved = False
            for i in range(N):
                # Try position perturbations
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        new_centers = [list(c) for c in best_centers]
                        new_centers[i][0] += dx
                        new_centers[i][1] += dy
                        if check_disjoint(new_centers, best_angles):
                            side = outer_side_fast(new_centers, best_angles, best_ocx, best_ocy, best_otheta)
                            if side < best_side - 1e-10:
                                best_side = side
                                best_centers = new_centers
                                improved = True
                # Try angle perturbations
                for da in [-15, 0, 15]:
                    new_angles = list(best_angles)
                    new_angles[i] = best_angles[i] + da
                    if check_disjoint(best_centers, new_angles):
                        side = outer_side_fast(best_centers, new_angles, best_ocx, best_ocy, best_otheta)
                        if side < best_side - 1e-10:
                            best_side = side
                            best_angles = new_angles
                            improved = True
            if not improved:
                # Try smaller step
                step *= 0.7
                if step < 0.001:
                    break
        
        return best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_side
    
    # Generate initial configurations
    best_result = None
    best_score = float('inf')
    
    # Configuration 1: Hexagonal lattice (compact)
    d = math.sqrt(3)  # distance between centers of touching hexagons
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    # Ring 0: center
    centers1 = [np.array([0.0, 0.0])]
    
    # Ring 1: 6 neighbors
    ring1 = [u, v, v-u, -u, -v, u-v]
    for vec in ring1:
        centers1.append(vec.copy())
    
    # Ring 2: pick 5 from ring 2 candidates
    ring2_cands = []
    basis = [u, v, v-u, -u, -v, u-v]
    for i in range(6):
        p1 = ring1[i]
        for b in basis:
            p2 = p1 + b
            # Check not in ring 0 or ring 1
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=0.01)
            is_r1 = any(np.allclose(p2, r, atol=0.01) for r in ring1)
            if not is_r0 and not is_r1:
                dup = any(np.allclose(p2, pc, atol=0.01) for pc in ring2_cands)
                if not dup:
                    ring2_cands.append(p2.copy())
    
    ring2_cands.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers1.append(ring2_cands[i].copy())
    
    assert len(centers1) == 12
    centers1_list = [[float(c[0]), float(c[1])] for c in centers1]
    angles1 = [30.0] * 12
    
    # Optimize configuration 1
    c1, a1, ocx1, ocy1, oth1, s1 = optimize(centers1_list, angles1, max_time=100)
    if s1 < best_score:
        best_score = s1
        best_result = (c1, a1, ocx1, ocy1, oth1)
    
    # Configuration 2: Different arrangement - 3 rows of 4
    # Try a more spread out configuration
    if time.time() < deadline - 40:
        centers2 = []
        # 4 in a row, offset
        for row in range(3):
            y_off = row * d * math.sqrt(3) / 2
            x_off = (row % 2) * d / 2
            for col in range(4):
                x = col * d + x_off
                y = y_off
                centers2.append([x, y])
        
        # Center them
        cx_mean = sum(c[0] for c in centers2) / 12
        cy_mean = sum(c[1] for c in centers2) / 12
        centers2 = [[c[0] - cx_mean, c[1] - cy_mean] for c in centers2]
        
        angles2 = [30.0] * 12
        
        c2, a2, ocx2, ocy2, oth2, s2 = optimize(centers2, angles2, max_time=40)
        if s2 < best_score:
            best_score = s2
            best_result = (c2, a2, ocx2, ocy2, oth2)
    
    # Configuration 3: Random restarts
    if time.time() < deadline - 30:
        for restart in range(10):
            if time.time() > deadline - 25:
                break
            # Random positions in a circle
            centers3 = []
            for i in range(12):
                r = random.uniform(0, 3.5)
                theta = random.uniform(0, 2*math.pi)
                centers3.append([r*math.cos(theta), r*math.sin(theta)])
            angles3 = [random.choice([0, 30, 60, 90, 120, 150]) for _ in range(12)]
            
            # Check disjointness, if not, skip
            if not check_disjoint(centers3, angles3):
                continue
            
            c3, a3, ocx3, ocy3, oth3, s3 = optimize(centers3, angles3, max_time=2.5)
            if s3 < best_score:
                best_score = s3
                best_result = (c3, a3, ocx3, ocy3, oth3)
    
    # Final optimization on best result
    if time.time() < deadline - 5:
        c, a, ocx, ocy, oth, s = optimize(best_result[0], best_result[1], max_time=15)
        if s < best_score:
            best_score = s
            best_result = (c, a, ocx, ocy, oth)
    
    # Format output
    inner = [[float(c[0]), float(c[1]), float(a)] for c, a in zip(best_result[0], best_result[1])]
    outer_center = (float(best_result[2]), float(best_result[3]))
    outer_side = float(best_score)
    outer_angle = float(best_result[4])
    
    return inner, outer_center, outer_side, outer_angle
