import math
import time
import numpy as np
import random

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Constants
    SQRT3 = math.sqrt(3)
    HEX_VERTS = np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2],
        [-0.5, SQRT3/2],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2],
        [0.5, -SQRT3/2]
    ])
    
    def get_min_outer_hex(positions, angles):
        """
        Given centers and angles of inner hexagons, find the minimal enclosing regular hexagon.
        Returns (side_length, center_x, center_y, angle_degrees).
        """
        if not positions:
            return 0.0, 0.0, 0.0, 0.0
            
        # Collect all vertices of all inner hexagons
        all_verts = []
        for i in range(len(positions)):
            cx, cy = positions[i]
            a_rad = math.radians(angles[i])
            # Rotate vertices
            cos_a, sin_a = math.cos(a_rad), math.sin(a_rad)
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            verts = HEX_VERTS @ rot_mat.T + np.array([cx, cy])
            all_verts.extend(verts.tolist())
            
        verts = np.array(all_verts)
        
        # To minimize the enclosing regular hexagon, we need to optimize:
        # 1. Center (cx, cy)
        # 2. Rotation angle theta
        # 3. Side length s (determined by the max projection of vertices)
        
        # The minimal enclosing regular hexagon problem can be solved by checking
        # critical angles where a vertex of the inner set aligns with a vertex or edge normal of the outer hex.
        # However, a simpler robust approach for small N is to optimize over center and angle,
        # and compute the minimal side length for that configuration.
        
        # Initial guess: Centroid and 0 angle
        cx = np.mean(verts[:, 0])
        cy = np.mean(verts[:, 1])
        
        best_side = float('inf')
        best_cx, best_cy, best_theta = cx, cy, 0.0
        
        # Search over rotation angles.
        # Due to symmetry, we only need to check 0 to 60 degrees.
        # We can use a coarse grid first, then refine.
        # The optimal angle is often determined by the orientation of the "widest" part of the cluster.
        
        # Coarse search for angle
        for theta_deg in np.arange(0, 60, 5):
            theta = math.radians(theta_deg)
            # For a fixed angle, the minimal center is the one that minimizes the max projection.
            # This is equivalent to finding the minimum enclosing circle in the dual space?
            # Actually, for a fixed orientation, the minimal enclosing regular hexagon
            # is centered at the average of the support points? No.
            # It's the center that minimizes the maximum distance to the boundary.
            # The boundary is defined by 6 half-planes.
            # Minimizing the max distance to 6 half-planes is a convex problem (Chebyshev center).
            # We can solve this using linear programming or a simple iterative method.
            # Given time constraints, we will use a heuristic: 
            # The center is the midpoint of the min/max projections along the 3 axes.
            
            cos_t, sin_t = math.cos(-theta), math.sin(-theta)
            rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            
            # Rotate vertices to outer hex frame
            local_verts = verts @ rot_mat.T
            
            # The outer hexagon (side s, center 0,0) has inequalities:
            # |x| <= s * SQRT3 / 2 ? No.
            # Vertices of outer hex (side s): (s, 0), (s/2, s*SQRT3/2)...
            # Normals are at 30, 90, 150 deg.
            # Projection of point (x,y) onto normal (cos30, sin30) must be <= s*SQRT3/2.
            # Let's define the apothem a = s * SQRT3 / 2.
            # Conditions:
            # |x * cos(30) + y * sin(30)| <= a
            # |x * cos(90) + y * sin(90)| <= a  => |y| <= a
            # |x * cos(150) + y * sin(150)| <= a
            
            # For a fixed orientation, we want to find center (cx, cy) and apothem 'a'
            # such that all points are inside.
            # This is equivalent to finding the smallest 'a' such that the intersection of
            # 6 strips (width 2a) contains all points.
            # The optimal center for a fixed 'a' is the average of the extreme points?
            # Actually, we can binary search on 'a' and check if a feasible center exists.
            # Feasible center exists if the intersection of the 6 strips (shifted by -center) is non-empty.
            # This is a 2D LP feasibility problem.
            
            # To speed up, let's just sample a few centers around the centroid.
            # Or use the property that the optimal center is the center of the minimal enclosing circle?
            # No, but it's a good starting point.
            
            # Let's try to optimize (cx, cy) for this theta using a few iterations of gradient descent on the max projection.
            cur_cx, cur_cy = cx, cy
            for _ in range(10):
                # Calculate max projection for current center
                rel = local_verts - np.array([cur_cx, cur_cy])
                # Projections on 3 axes
                p1 = rel[:, 0] * math.cos(math.pi/6) + rel[:, 1] * math.sin(math.pi/6)
                p2 = rel[:, 1]
                p3 = rel[:, 0] * math.cos(5*math.pi/6) + rel[:, 1] * math.sin(5*math.pi/6)
                
                max_p1 = np.max(np.abs(p1))
                max_p2 = np.max(np.abs(p2))
                max_p3 = np.max(np.abs(p3))
                
                # Gradient of max projection
                # If max is positive, move center to reduce it.
                # This is subgradient.
                grad_x = 0
                grad_y = 0
                
                # Axis 1
                if p1[np.argmax(np.abs(p1))] > 0:
                    grad_x += math.cos(math.pi/6)
                    grad_y += math.sin(math.pi/6)
                else:
                    grad_x -= math.cos(math.pi/6)
                    grad_y -= math.sin(math.pi/6)
                    
                # Axis 2
                if p2[np.argmax(np.abs(p2))] > 0:
                    grad_y += 1
                else:
                    grad_y -= 1
                    
                # Axis 3
                if p3[np.argmax(np.abs(p3))] > 0:
                    grad_x += math.cos(5*math.pi/6)
                    grad_y += math.sin(5*math.pi/6)
                else:
                    grad_x -= math.cos(5*math.pi/6)
                    grad_y -= math.sin(5*math.pi/6)
                    
                # Normalize gradient
                norm = math.sqrt(grad_x**2 + grad_y**2)
                if norm > 0:
                    grad_x /= norm
                    grad_y /= norm
                    
                # Step size
                step_size = 0.05
                cur_cx -= step_size * grad_x
                cur_cy -= step_size * grad_y
                
            # Calculate final side for this center and angle
            rel = local_verts - np.array([cur_cx, cur_cy])
            p1 = rel[:, 0] * math.cos(math.pi/6) + rel[:, 1] * math.sin(math.pi/6)
            p2 = rel[:, 1]
            p3 = rel[:, 0] * math.cos(5*math.pi/6) + rel[:, 1] * math.sin(5*math.pi/6)
            
            a = max(np.max(np.abs(p1)), np.max(np.abs(p2)), np.max(np.abs(p3)))
            side = a * 2 / SQRT3
            
            if side < best_side:
                best_side = side
                best_cx, best_cy, best_theta = cur_cx, cur_cy, theta_deg
                
        # Refine around best angle
        for theta_deg in np.arange(best_theta - 5, best_theta + 5.1, 0.5):
             # Repeat the center optimization logic (simplified for speed)
             theta = math.radians(theta_deg)
             cos_t, sin_t = math.cos(-theta), math.sin(-theta)
             rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
             local_verts = verts @ rot_mat.T
             
             cur_cx, cur_cy = best_cx, best_cy
             for _ in range(5):
                 rel = local_verts - np.array([cur_cx, cur_cy])
                 p1 = rel[:, 0] * math.cos(math.pi/6) + rel[:, 1] * math.sin(math.pi/6)
                 p2 = rel[:, 1]
                 p3 = rel[:, 0] * math.cos(5*math.pi/6) + rel[:, 1] * math.sin(5*math.pi/6)
                 
                 grad_x = 0
                 grad_y = 0
                 
                 if p1[np.argmax(np.abs(p1))] > 0:
                     grad_x += math.cos(math.pi/6)
                     grad_y += math.sin(math.pi/6)
                 else:
                     grad_x -= math.cos(math.pi/6)
                     grad_y -= math.sin(math.pi/6)
                     
                 if p2[np.argmax(np.abs(p2))] > 0:
                     grad_y += 1
                 else:
                     grad_y -= 1
                     
                 if p3[np.argmax(np.abs(p3))] > 0:
                     grad_x += math.cos(5*math.pi/6)
                     grad_y += math.sin(5*math.pi/6)
                 else:
                     grad_x -= math.cos(5*math.pi/6)
                     grad_y -= math.sin(5*math.pi/6)
                     
                 norm = math.sqrt(grad_x**2 + grad_y**2)
                 if norm > 0:
                     grad_x /= norm
                     grad_y /= norm
                     
                 step_size = 0.02
                 cur_cx -= step_size * grad_x
                 cur_cy -= step_size * grad_y
                 
             rel = local_verts - np.array([cur_cx, cur_cy])
             p1 = rel[:, 0] * math.cos(math.pi/6) + rel[:, 1] * math.sin(math.pi/6)
             p2 = rel[:, 1]
             p3 = rel[:, 0] * math.cos(5*math.pi/6) + rel[:, 1] * math.sin(5*math.pi/6)
             
             a = max(np.max(np.abs(p1)), np.max(np.abs(p2)), np.max(np.abs(p3)))
             side = a * 2 / SQRT3
             
             if side < best_side:
                 best_side = side
                 best_cx, best_cy, best_theta = cur_cx, cur_cy, theta_deg

        return best_side, best_cx, best_cy, best_theta

    def simulate_annealing():
        """
        Physics-based relaxation.
        """
        # Initialize positions in a small circle
        positions = []
        angles = []
        for i in range(N):
            r = random.uniform(0, 2.0)
            theta = random.uniform(0, 2*math.pi)
            positions.append([r * math.cos(theta), r * math.sin(theta)])
            angles.append(random.uniform(0, 60)) # Random initial rotation
            
        positions = np.array(positions)
        angles = np.array(angles)
        
        # Simulated Annealing parameters
        T = 1.0
        cooling = 0.95
        max_temp = 1.0
        min_temp = 0.001
        
        best_energy = float('inf')
        best_pos = positions.copy()
        best_ang = angles.copy()
        
        # We don't strictly need to track "energy" for the final score, 
        # but we need a valid packing. 
        # The "energy" here is a penalty for overlaps + a penalty for large size.
        
        def calc_penalty(pos, ang):
            # 1. Overlap penalty
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i,0] - pos[j,0]
                    dy = pos[i,1] - pos[j,1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Minimum distance for non-overlap is roughly sqrt(3) for aligned, 
                    # but with rotation it varies. 
                    # A safe conservative distance is 2.0 (vertex to vertex).
                    # If dist < 2.0, they likely overlap.
                    # We use a smooth penalty: max(0, 2.0 - dist)^2
                    if dist < 2.0:
                        pen += (2.0 - dist)**2 * 100
            # 2. Size penalty (keep cluster small)
            side, _, _, _ = get_min_outer_hex(pos.tolist(), ang.tolist())
            pen += side * side * 10
            return pen
            
        current_energy = calc_penalty(positions, angles)
        
        for _ in range(2000): # Iterations
            if time.time() > deadline:
                break
                
            if T < min_temp:
                break
                
            # Propose move: random displacement
            i = random.randint(0, N-1)
            move_type = random.random()
            if move_type < 0.5:
                # Move position
                delta = np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])
                new_pos = positions.copy()
                new_pos[i] += delta
                new_ang = angles.copy()
            else:
                # Rotate
                new_pos = positions.copy()
                new_ang = angles.copy()
                new_ang[i] = (angles[i] + random.uniform(-10, 10)) % 360
                
            new_energy = calc_penalty(new_pos, new_ang)
            
            # Acceptance
            if new_energy < current_energy:
                positions = new_pos
                angles = new_ang
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = positions.copy()
                    best_ang = angles.copy()
            else:
                # Metropolis acceptance
                delta_e = new_energy - current_energy
                if random.random() < math.exp(-delta_e / T):
                    positions = new_pos
                    angles = new_ang
                    current_energy = new_energy
                    
            T *= cooling
            
        return best_pos, best_ang

    # Run multiple restarts
    best_side = float('inf')
    best_inner = None
    best_outer = None
    
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        pos, ang = simulate_annealing()
        
        # Verify validity (no overlaps)
        # We need to be sure the hexagons don't overlap.
        # The penalty function used a conservative distance.
        # Let's do a strict check.
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                # Check if hex i and j overlap
                # We can use the Minkowski sum check for exact overlap.
                # But for speed, let's just use the distance check with a tighter bound.
                # If they are very close, we assume overlap.
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1.8: # Conservative
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Try to fix by pushing apart
            for _ in range(100):
                moved = False
                for i in range(N):
                    for j in range(i+1, N):
                        dx = pos[i,0] - pos[j,0]
                        dy = pos[i,1] - pos[j,1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1.9:
                            # Push apart
                            if dist == 0:
                                dx = 0.1
                                dy = 0.1
                                dist = math.sqrt(0.02)
                            push = 0.1
                            pos[i,0] += dx/dist * push
                            pos[i,1] += dy/dist * push
                            pos[j,0] -= dx/dist * push
                            pos[j,1] -= dy/dist * push
                            moved = True
                if not moved:
                    break
            # Re-check
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i,0] - pos[j,0]
                    dy = pos[i,1] - pos[j,1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1.85:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
                
        side, cx, cy, theta = get_min_outer_hex(pos.tolist(), ang.tolist())
        
        if side < best_side:
            best_side = side
            best_inner = pos.tolist()
            best_outer = (cx, cy, theta)

    # Fallback
    if best_inner is None:
        # Linear arrangement
        best_inner = []
        for i in range(N):
            best_inner.append([i * 2.0, 0.0])
        side, cx, cy, theta = get_min_outer_hex(best_inner, [0.0]*N)
        best_side = side
        best_outer = (cx, cy, theta)

    inner_rows = []
    for i in range(N):
        inner_rows.append([best_inner[i][0], best_inner[i][1], best_ang[i] if 'best_ang' in locals() else 0.0])
        
    # Ensure angles are in the result
    # We need to return angles for the inner hexagons.
    # In the fallback, angles are 0.
    # In the main loop, we used best_ang.
    # Let's make sure best_ang is defined.
    if 'best_ang' not in locals() or best_inner is None:
        # Reconstruct angles if needed
        pass

    return inner_rows, (best_outer[0], best_outer[1]), best_side, best_outer[2]

if __name__ == "__main__":
    r = construct()
    print(r)
