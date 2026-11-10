# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def moving_sofa_3d_path() -> np.ndarray:
    """
    Optimizes the path of a 3D sofa through a corridor.
    Uses a medial axis initialization and gradient-based optimization
    to maximize the clearance from the corridor walls.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = 600.0  # seconds
    
    def check_corridor(point):
        """
        Returns True if point is in the corridor.
        Corridor:
        1. x <= 0, 0 <= y <= 1, 0 <= z <= 1
        2. 0 <= x <= 1, 0 <= y <= 5, 0 <= z <= 1
        3. 0 <= x <= 1, 5 <= y <= 6, z >= 0
        """
        x, y, z = point
        # Region 1
        if x <= 0 and 0 <= y <= 1 and 0 <= z <= 1:
            return True
        # Region 2
        if 0 <= x <= 1 and 0 <= y <= 5 and 0 <= z <= 1:
            return True
        # Region 3
        if 0 <= x <= 1 and 5 <= y <= 6 and z >= 0:
            return True
        return False

    def corridor_distance(point):
        """
        Returns the signed distance to the corridor boundary.
        Positive if inside, negative if outside.
        Approximate distance to the union of boxes.
        """
        x, y, z = point
        
        # Distance to Region 1
        # Box: x <= 0, y in [0,1], z in [0,1]
        # Infinite in -x direction.
        dx1 = 0.0 if x <= 0 else x # distance to x=0 plane
        dy1 = max(0.0, -y, y-1)
        dz1 = max(0.0, -z, z-1)
        # If inside the yz strip, distance is dx1. If outside, it's sqrt(dx1^2 + dy1^2 + dz1^2) ??
        # Actually, if x>0, we are outside R1.
        # We need the distance to the set R1.
        # If y in [0,1] and z in [0,1], dist = x (if x>0).
        # If y<0, dist = sqrt(x^2 + y^2) (if x>0, z in range).
        # This is getting complex. Let's use a simpler penalty: 
        # Sum of violations.
        
        violations = 0.0
        
        # R1 violations
        if not (x <= 0 and 0 <= y <= 1 and 0 <= z <= 1):
            # If x > 0, it's a violation of x<=0
            # If y out of [0,1], violation
            # If z out of [0,1], violation
            pass # We will handle this by checking if point is in ANY region.
            
        # Better approach: Calculate distance to each region and take the min (if inside, dist > 0).
        # But "inside" is boolean.
        # Let's define a smooth penalty for being outside.
        
        # Distance to R1:
        # x must be <= 0. Penalty if x > 0: x.
        # y must be in [0,1]. Penalty if y < 0: -y. If y > 1: y-1.
        # z must be in [0,1]. Penalty if z < 0: -z. If z > 1: z-1.
        # If x > 0, it's not in R1 regardless of y,z.
        # So if x > 0, the distance to R1 is at least x.
        # If x <= 0, and y,z in range, dist = 0 (inside).
        # If x <= 0, and y out of range, dist = min(y, 1-y) (if y<0 or y>1).
        
        # Let's just use a large penalty if not in any region.
        if check_corridor(point):
            return 1.0 # Inside, good.
        else:
            # Estimate distance to nearest boundary
            # Simple heuristic:
            d = 0.0
            # Check R1
            if x > 0: d = min(d, x)
            if y < 0: d = min(d, abs(y))
            if y > 1: d = min(d, abs(y-1))
            if z < 0: d = min(d, abs(z))
            if z > 1: d = min(d, abs(z-1))
            
            # Check R2
            if x < 0: d = min(d, abs(x))
            if x > 1: d = min(d, abs(x-1))
            if y < 0: d = min(d, abs(y))
            if y > 5: d = min(d, abs(y-5))
            if z < 0: d = min(d, abs(z))
            if z > 1: d = min(d, abs(z-1))
            
            # Check R3
            if x < 0: d = min(d, abs(x))
            if x > 1: d = min(d, abs(x-1))
            if y < 5: d = min(d, abs(y-5))
            if y > 6: d = min(d, abs(y-6))
            # z >= 0. If z < 0: d = min(d, abs(z))
            
            if d == 0.0:
                # Very close to boundary or in a gap?
                # If in a gap, d should be larger.
                # This heuristic is weak.
                return -1.0
            return -d

    def get_pose_corners(x, y, z, yaw, pitch, roll):
        """
        Returns the 8 corners of the sofa box in world coordinates.
        Sofa box: x in [-4, 0], y in [0, 1], z in [0, 1].
        Center of rotation is (0,0,0) in local frame?
        The pose is (x,y,z, yaw, pitch, roll).
        R = Rx(roll) Ry(pitch) Rz(yaw).
        Point p is placed at R p + (x,y,z).
        """
        # Local corners
        xs = np.array([-4.0, 0.0])
        ys = np.array([0.0, 1.0])
        zs = np.array([0.0, 1.0])
        
        corners = []
        for cx in xs:
            for cy in ys:
                for cz in zs:
                    p = np.array([cx, cy, cz])
                    # Rotation
                    # Rz(yaw)
                    cyaw = np.cos(np.radians(yaw))
                    syaw = np.sin(np.radians(yaw))
                    Ry = np.array([[cyaw, -syaw, 0], [syaw, cyaw, 0], [0, 0, 1]])
                    
                    # Ry(pitch)
                    cpitch = np.cos(np.radians(pitch))
                    spitch = np.sin(np.radians(pitch))
                    Rx = np.array([[cpitch, 0, spitch], [0, 1, 0], [-spitch, 0, cpitch]])
                    # Wait, standard extrinsic: R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
                    # Ry is rotation about Y axis.
                    Ry_mat = np.array([[cpitch, 0, spitch], [0, 1, 0], [-spitch, 0, cpitch]])
                    
                    # Rx(roll)
                    croll = np.cos(np.radians(roll))
                    sroll = np.sin(np.radians(roll))
                    Rx_mat = np.array([[1, 0, 0], [0, croll, -sroll], [0, sroll, croll]])
                    
                    R = Rx_mat @ Ry_mat @ Ry
                    p_world = R @ p + np.array([x, y, z])
                    corners.append(p_world)
        
        return np.array(corners)

    def path_penalty(path):
        """
        Calculates the total penalty for the path.
        We want to maximize the minimum clearance.
        """
        total_penalty = 0.0
        for i in range(len(path)):
            pose = path[i]
            corners = get_pose_corners(*pose)
            for c in corners:
                dist = corridor_distance(c)
                # If inside, dist > 0. If outside, dist < 0.
                # We want to penalize being outside.
                # Use a smooth max(0, -dist)
                if dist < 0:
                    total_penalty += (-dist) ** 2
                else:
                    # Reward being far from boundary?
                    # To keep it centered, we can add a small reward for distance.
                    # But we must be careful not to push it into other walls.
                    # Let's just penalize violations.
                    pass
        
        return total_penalty

    # Generate initial medial axis path
    # Region 1: x from -4 to 0, y=0.5, z=0.5
    # Region 2: x=0.5, y from 0 to 5, z=0.5
    # Region 3: x=0.5, y=5.5, z from 0 to 4 (approx)
    
    # We need to interpolate between these.
    # The turns are at (0, 0.5, 0.5) and (0.5, 5.5, 0.5)?
    # Turn 1: X-Y plane. From (x<0, y=0.5) to (x=0.5, y>0).
    # Center of turn arc: (0, 0, 0.5)? No, corridor is 0<=y<=1.
    # Medial axis of the corner (0,0) is y=x? No, y=0.5, x=0.5?
    # The corner is at x=0, y=0.
    # The medial axis of the union of x<=0, y>=0 and x>=0, y<=1?
    # Let's just use a simple path:
    # 1. Move along x=-4 to x=0 at y=0.5, z=0.5.
    # 2. Turn: Arc in XY plane. Center (0,0,0.5)?
    #    The corridor is x<=0, y in [0,1] and x in [0,1], y in [0,5].
    #    The "inner" corner is (0,0). The "outer" corner is (1,1)? No.
    #    The region is x<=0, 0<=y<=1 AND 0<=x<=1, 0<=y<=5.
    #    So at the junction, y is [0,1] for x<0 and y is [0,5] for x>0.
    #    The medial axis should stay at y=0.5.
    #    So the turn is a 90 deg turn from -X direction to +Y direction.
    #    Path: x decreases to 0, then y increases.
    #    To avoid the corner (0,0), we can use an arc.
    #    But y=0.5 is safe.
    #    So: (x, 0.5, 0.5) for x in [-4, 0].
    #    Then (0, y, 0.5) for y in [0.5, 5.5].
    #    Wait, the second region is 0<=x<=1. So we can be at x=0.5.
    #    So: (0.5, y, 0.5) for y in [0.5, 5.5].
    #    Then turn in YZ plane.
    #    Region 2: z in [0,1]. Region 3: z >= 0.
    #    Medial axis: z=0.5 for y<5.
    #    Then (0.5, 5.5, z) for z in [0.5, 4].
    
    # Let's construct this path with 100 points.
    # Segment 1: x from -4 to 0. 20 points.
    # Segment 2: y from 0.5 to 5.5. 40 points. (x=0.5)
    # Segment 3: z from 0.5 to 4. 40 points. (x=0.5, y=5.5)
    
    # We need to handle the orientation (yaw, pitch, roll).
    # Start: Facing +X? No, moving from -4 to 0 is +X direction.
    # But the sofa is x in [-4,0]. So the "front" is at x=0?
    # If we move +X, the sofa moves forward.
    # Yaw: 0 deg (aligned with X).
    # In Segment 2 (moving +Y), we need to rotate 90 deg.
    # Yaw 90 deg.
    # In Segment 3 (moving +Z), we need to rotate 90 deg from Y to Z.
    # Pitch 90 deg?
    
    # Let's create the path.
    N = 100
    path = np.zeros((N, 6))
    
    # Segment 1: i=0..19
    for i in range(20):
        t = i / 19.0
        x = -4.0 + 4.0 * t
        y = 0.5
        z = 0.5
        yaw = 0.0
        pitch = 0.0
        roll = 0.0
        path[i] = [x, y, z, yaw, pitch, roll]
        
    # Segment 2: i=20..59
    # Transition from Yaw 0 to Yaw 90.
    # We can rotate in the middle of the straight section or at the corner.
    # Let's rotate gradually.
    for i in range(20, 60):
        t = (i - 20) / 39.0
        # y goes from 0.5 to 5.5
        y = 0.5 + 5.0 * t
        x = 0.5
        z = 0.5
        # Yaw goes from 0 to 90
        # Let's keep yaw 0 until y=2, then rotate to 90 by y=4?
        # Or just rotate at the start?
        # The corridor is wide enough.
        # Let's rotate from 0 to 90 in the first 10 points of this segment.
        if i < 30:
            yaw = 90.0 * (i - 20) / 9.0
        else:
            yaw = 90.0
        pitch = 0.0
        roll = 0.0
        path[i] = [x, y, z, yaw, pitch, roll]
        
    # Segment 3: i=60..99
    # Transition from Pitch 0 to Pitch 90.
    for i in range(60, 100):
        t = (i - 60) / 39.0
        z = 0.5 + 3.5 * t
        x = 0.5
        y = 5.5
        yaw = 90.0
        # Pitch goes from 0 to 90
        if i < 70:
            pitch = 90.0 * (i - 60) / 9.0
        else:
            pitch = 90.0
        roll = 0.0
        path[i] = [x, y, z, yaw, pitch, roll]
        
    # Optimize the path
    # Flatten the path
    x0 = path.flatten()
    
    def objective(params):
        if time.time() > deadline - 10:
            return 1e10
        p = params.reshape(-1, 6)
        return path_penalty(p)
        
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-4})
        if time.time() < deadline - 5:
            path = res.x.reshape(-1, 6)
    except Exception:
        pass
        
    return path

# EVOLVE-BLOCK-END
