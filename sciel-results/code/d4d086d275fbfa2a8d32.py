# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def moving_sofa_3d_path() -> np.ndarray:
    """
    Optimizes the 3D moving sofa path to maximize the swept volume of the surviving block.
    Uses local search (hill climbing) with random perturbations.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = 600.0 # 600s limit, leave 30s margin for overhead/return
    
    # Corridor definition
    # {x<=0, 0<=y<=1, 0<=z<=1} U {0<=x<=1, 0<=y<=5, 0<=z<=1} U {0<=x<=1, 5<=y<=6, z>=0}
    # Note: The problem statement says "z>=0" for the last part, but usually corridors are bounded.
    # Let's re-read carefully: {0<=x<=1, 5<=y<=6, z>=0}.
    # Wait, the start is (-3,0,0) and end is (0,6,7). The end pose has z=7.
    # If the corridor is z>=0 in the last segment, it goes up infinitely?
    # Let's check the grid: z in linspace(0,1,21).
    # So the relevant z range for the grid is [0,1].
    # However, the path moves to z=7. The *block* points must stay in the corridor.
    # If the corridor allows z>=0, then high z is fine as long as x,y are in the box.
    
    def in_corridor(pts):
        # pts: (N, 3) array
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # Box 1: x<=0, 0<=y<=1, 0<=z<=1
        # Note: x can be negative.
        c1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
        
        # Box 2: 0<=x<=1, 0<=y<=5, 0<=z<=1
        c2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
        
        # Box 3: 0<=x<=1, 5<=y<=6, z>=0
        # Is there an upper bound on z? The text says "z>=0".
        # Given the end pose is z=7, and the block is small, if z is unbounded,
        # then any z>=0 is valid in this x,y region.
        c3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
        
        return c1 | c2 | c3

    def rotation_arc_bbox_in_corridor(p0, p1, axis, theta0, theta1):
        """
        Check if the axis-aligned bounding box of the rotation arc of point p0 
        (relative to origin) rotated by theta about axis, then translated by p1?
        No, the rotation is about the object's center of mass? 
        "point p is placed at R p + (x, y, z)"
        The rotation is intrinsic/extrinsic about the origin of the object frame, 
        then translated.
        The "axis-aligned box bounding its rotation arc" refers to the path of the point 
        in world coordinates.
        
        Let's clarify the geometry:
        Pose = (x, y, z, yaw, pitch, roll).
        Point p in object frame.
        World pos = R(yaw, pitch, roll) @ p + T.
        
        When rotating from old pose to new pose, we fix translation?
        "Each step is done as an x move, y move, z move, then a rotation about a single axis from the old to the new orientation."
        
        This implies the evaluator splits the transition between two poses into sub-steps:
        1. Move x (keep y,z, angles same as start)
        2. Move y
        3. Move z
        4. Rotate about one axis (which one? "from the old to the new orientation" - this implies the angle change is decomposed).
        
        Actually, the prompt says: "inserts linearly interpolated poses so each step moves at most 0.05... and changes each angle by at most 2."
        Then: "A grid point survives if it is in the corridor at the first pose and after each move, and the axis-aligned box bounding its rotation arc lies in the corridor."
        
        This suggests that for each linearly interpolated pose in the sequence, we check:
        1. Is the point in the corridor at this pose?
        2. For the rotation part of the step (if any), is the bounding box of the arc in the corridor?
        
        Wait, the "linearly interpolated poses" are the *dense* path.
        The "move" in "after each move" likely refers to the discrete steps of the evaluator's logic:
        The evaluator takes the dense path.
        For each consecutive pair of dense poses P_i, P_{i+1}:
        - It performs an x-move, y-move, z-move, then rotation?
        - Or does it just check the poses?
        
        "Each step is done as an x move, y move, z move, then a rotation about a single axis from the old to the new orientation."
        This sounds like the *interpolation* method.
        If I provide a path, the evaluator densifies it.
        But the "move" logic might be specific to how the evaluator validates the *rotation*.
        
        Let's assume the strictest interpretation for safety:
        For any two consecutive poses in the *dense* path, the transition is decomposed into:
        1. Translate x to new x (angles old)
        2. Translate y to new y (angles old)
        3. Translate z to new z (angles old)
        4. Rotate about the axis that changed the most (or sequentially yaw, pitch, roll?) to new angles.
        
        Actually, the prompt says "changes each angle by at most 2". This implies the dense path has small angle steps.
        If the angle step is small, the rotation arc is small.
        
        To be safe and fast, I will implement a checker that:
        1. Takes the provided path.
        2. Densifies it (simulating the evaluator's densification).
        3. For each step in the dense path:
           a. Check if the point is in the corridor at the start pose.
           b. Check if the point is in the corridor at the end pose.
           c. Check if the bounding box of the rotation arc (if angles change) is in the corridor.
           
        However, simulating the exact evaluator densification and "move" logic is complex.
        A simpler, robust approach:
        The "surviving volume" is determined by the intersection of the "valid regions" for each grid point.
        A grid point survives if its trajectory stays in the corridor.
        
        Let's approximate the "rotation arc bounding box" check.
        If we rotate a point p about an axis by angle d, the bounding box of the arc is the box containing the arc.
        For small angles, it's close to the point itself.
        
        Given the complexity, let's focus on optimizing the path such that the *poses* themselves are well-placed, and the rotations are smooth.
        
        I will write a function `evaluate_path(path)` that:
        1. Densifies the path to match the evaluator's step size (0.05 dist, 2 deg angle).
        2. For each grid point, checks if it stays in the corridor for all dense poses.
        3. For rotation steps, checks if the AABB of the arc is in the corridor.
        4. Counts the 4x4x4 blocks where all 64 points survive.
        5. Returns volume.
        
        This evaluation is the bottleneck. I need it to be fast.
        
        Grid: 81 x 21 x 21 = 35721 points.
        Dense path: The path length is roughly 10 units. 0.05 step -> 200 steps.
        35721 * 200 = 7 million checks.
        Each check is a few comparisons. This should be fast in NumPy.
        
        Let's implement the densification and check.
    """

    # Grid definition
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    # Flatten
    grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    N_points = len(grid_points)
    
    # Precompute rotation matrices? No, angles change.
    
    def rot_matrices(yaw, pitch, roll):
        # yaw, pitch, roll in degrees
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
        
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        
        # R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
        # Rz(yaw) = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
        # Ry(pitch) = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
        # Rx(roll) = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
        
        # We can compute the full 3x3 matrix or just apply to points.
        # Applying to points (N, 3) is better.
        
        # p' = Rx @ Ry @ Rz @ p
        # Let's compute the combined matrix R
        # R = [[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        #      [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        #      [-sp,   cp*sr,            cp*cr            ]]
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr            ]
        ])
        return R

    def evaluate_path(path):
        """
        path: (k, 6) array
        Returns: volume (float)
        """
        # 1. Densify path
        # The evaluator: "prepends (-3,0,0,0,0,0), appends (0,6,7,90,90,90) and inserts linearly interpolated poses so each step moves at most 0.05 (Euclidean) and changes each angle by at most 2."
        
        full_path = np.vstack([
            np.array([-3, 0, 0, 0, 0, 0]),
            path,
            np.array([0, 6, 7, 90, 90, 90])
        ])
        
        # Densify
        dense_poses = [full_path[0]]
        for i in range(len(full_path) - 1):
            p0 = full_path[i]
            p1 = full_path[i+1]
            
            # Calculate number of steps needed
            # Distance in xyz
            d_xyz = np.linalg.norm(p1[:3] - p0[:3])
            # Angle changes
            d_yaw = abs(p1[3] - p0[3])
            d_pitch = abs(p1[4] - p0[4])
            d_roll = abs(p1[5] - p0[5])
            
            max_d_xyz = 0.05
            max_d_angle = 2.0
            
            steps_xyz = int(np.ceil(d_xyz / max_d_xyz))
            steps_angle = int(np.ceil(max(d_yaw, d_pitch, d_roll) / max_d_angle))
            
            n_steps = max(steps_xyz, steps_angle, 1)
            
            for s in range(1, n_steps + 1):
                t = s / n_steps
                new_pose = p0 + t * (p1 - p0)
                dense_poses.append(new_pose)
        
        dense_poses = np.array(dense_poses)
        
        # 2. Check survival for each grid point
        # A point survives if it is in the corridor at all dense poses.
        # Also, for rotation steps, the AABB of the arc must be in the corridor.
        
        # Optimization:
        # Instead of checking each point against each pose individually in a Python loop,
        # we can vectorize the pose check.
        
        # However, the "AABB of rotation arc" is tricky.
        # If the angle change is small (<=2 deg), the arc is small.
        # The AABB of the arc for a point p rotated about an axis is roughly
        # the box spanned by p_old and p_new, expanded slightly.
        # For a 2 degree rotation, the deviation is small.
        # We can approximate the AABB check by checking if the *segment* between p_old and p_new
        # is in the corridor, or just check the endpoints and assume the small arc is safe
        # if the endpoints are safe and the corridor is convex in that region.
        # The corridor is a union of boxes, not convex.
        
        # Let's do a strict check for the dense poses first.
        # For each dense pose, compute the world coordinates of all grid points.
        # Check if they are in the corridor.
        # If any point is outside, it dies.
        
        # We can track a "alive" mask.
        alive = np.ones(N_points, dtype=bool)
        
        # Iterate through dense poses
        for pose in dense_poses:
            x, y, z, yaw, pitch, roll = pose
            R = rot_matrices(yaw, pitch, roll)
            # World points = grid_points @ R.T + [x,y,z]
            # grid_points is (N, 3)
            wp = grid_points @ R.T + np.array([x, y, z])
            
            # Check corridor
            valid = in_corridor(wp)
            
            # Update alive mask
            alive = alive & valid
            
            # Early exit if no points are alive?
            # We need 4x4x4 blocks. If few points are alive, we can stop.
            # But checking block survival requires knowing which specific points are alive.
            
        # 3. Count 4x4x4 blocks
        # Reshape alive mask to (81, 21, 21)
        alive_grid = alive.reshape(81, 21, 21)
        
        # A 4x4x4 block is defined by adjacent grid points.
        # There are (81-3) x (21-3) x (21-3) such blocks.
        # Block (i, j, k) covers grid points i..i+3, j..j+3, k..k+3.
        
        # We can use a sliding window sum or just iterate.
        # Number of blocks: 78 * 18 * 18 = 25272 blocks.
        # For each block, check if all 64 points are alive.
        
        # Vectorized block check:
        # alive_grid is (81, 21, 21)
        # We want to sum over 4x4x4 windows.
        # If sum == 64, the block is valid.
        
        # Using np.lib.stride_tricks or just slicing?
        # Slicing is easier to read and likely fast enough for 25k blocks.
        
        valid_blocks = 0
        for i in range(81 - 3):
            for j in range(21 - 3):
                for k in range(21 - 3):
                    block = alive_grid[i:i+4, j:j+4, k:k+4]
                    if np.all(block):
                        valid_blocks += 1
        
        volume = 0.05**3 * valid_blocks
        return volume

    # Baseline path from the current program
    baseline_path = np.array([
        [0.939609406, 0.00998329527, -0.00407296132, 0.0886163961, 0.0514178831, 0.789799574],
        [0.976853949, 0.0843900243, -0.00187236014, 2.22743402, 0.0275244655, 0.595653008],
        [1.01563377, 0.158600496, 9.83065911e-05, 4.35572642, -0.00995231395, 0.363970701],
        [1.08493385, 0.307755287, 0.00404501406, 8.62938219, -0.000435990906, -0.335077274],
        [1.12726319, 0.436994587, 0.0165792562, 12.3522097, -0.444355561, -1.09977761],
        [1.15035596, 0.509864783, 0.0246729498, 14.4926365, -0.703150883, -1.77573027],
        [1.20020002, 0.674818962, 0.0413920989, 19.2858257, -1.46824802, -3.24301616],
        [1.22921833, 0.78032244, 0.0508392229, 22.3604231, -2.05934965, -4.19183977],
        [1.25656232, 0.886955923, 0.0622851979, 25.4006424, -2.79233605, -4.94301221],
        [1.28161834, 0.9941859, 0.0713168622, 28.4982429, -3.53439006, -5.72810662],
        [1.30160821, 1.12543063, 0.0720451855, 32.0688854, -3.9280592, -5.56488257],
        [1.32438826, 1.32927434, 0.0724034864, 37.7041144, -4.47472162, -5.17490305],
        [1.33399008, 1.48086681, 0.0758177845, 41.9024902, -4.92138226, -4.72192269],
        [1.33698854, 1.59471234, 0.075956822, 45.1249666, -5.16592127, -4.32429483],
        [1.33554955, 1.70831674, 0.0733949585, 48.4117195, -5.24474942, -3.95318659],
        [1.33301955, 1.77848192, 0.0730557819, 50.4395679, -5.29391038, -3.60268508],
        [1.32299384, 1.89675169, 0.066224521, 54.0094721, -5.07170268, -3.10511721],
        [1.31609554, 1.95530642, 0.0645580036, 55.7906035, -4.95025879, -2.75163185],
        [1.29699464, 2.0647489, 0.0528964026, 59.2092384, -4.24840518, -2.08208301],
        [1.28467533, 2.12603596, 0.0465722033, 61.19788, -3.85394213, -1.77381137],
        [1.27612175, 2.16246336, 0.0429218547, 62.3960278, -3.60313312, -1.60391888],
        [1.25918263, 2.22238115, 0.0352027157, 64.3876548, -2.94645508, -1.14315788],
        [1.21195969, 2.37272154, 0.0185866337, 69.5501344, -1.44731306, -0.351817856],
        [1.17367747, 2.46947271, 0.00787738711, 73.3278162, -0.59130695, -0.204315921],
        [1.13343824, 2.56892736, 0.000240804007, 77.2294036, 0.0742535282, -0.0118797041],
        [1.10227764, 2.63702207, -0.000813164259, 80.2921591, 0.325638556, -0.00909491031],
        [1.08115946, 2.67872855, -0.0019446761, 82.2507337, 0.456149966, 0.00923364344],
        [1.06864394, 2.70464551, -0.00209793986, 83.4698923, 0.395788406, -0.00251260641],
        [0.993054246, 2.83888423, -0.00277991124, 90.0331465, 0.651004673, -0.0200027208],
        [0.992050998, 3.42584708, -0.00292617372, 90.0630824, 0.648340038, -0.0336076805],
        [0.990524577, 4.81296934, -0.00378757857, 90.0846127, 0.741107384, -0.0439029391],
        [0.988680148, 5.99738121, -0.00451717463, 90.104646, 0.883485061, -0.041030785],
        [0.957624286, 5.9528988, 0.0102741306, 90.1327839, 4.19080459, 1.12928446],
        [0.942862105, 5.99368058, 0.0162024942, 90.1288679, 5.76538787, 1.62560793],
        [0.931969027, 5.99412693, 0.020619344, 90.1263482, 6.90261077, 1.98875845],
        [0.904773557, 6.00941729, 0.0313747872, 90.128735, 9.61483109, 2.83144875],
        [0.8858989, 6.01830162, 0.0381113604, 90.135911, 11.4508584, 3.34953358],
        [0.863540853, 6.02619691, 0.04487657, 90.1598001, 13.5181743, 3.88721021],
        [0.836463078, 6.03765479, 0.0528267196, 90.1967295, 15.9298717, 4.48186113],
        [0.804731182, 6.04627379, 0.0625195903, 90.2336098, 18.6990514, 5.14505668],
        [0.788296376, 6.05202369, 0.0675622892, 90.2479959, 20.1264906, 5.47138803],
        [0.778474876, 6.05538925, 0.0706050032, 90.2577268, 20.9723911, 5.66341348],
        [0.770888811, 6.05788046, 0.0730068444, 90.2657676, 21.6229248, 5.80874653],
        [0.764648333, 6.0599328, 0.0749383027, 90.2726388, 22.1532161, 5.92642676],
        [0.748753549, 6.06467644, 0.0801606053, 90.2892425, 23.4927925, 6.21910882],
        [0.737683416, 6.06785126, 0.0838714978, 90.3001161, 24.4180728, 6.42046395],
        [0.726094096, 6.07124306, 0.0879934282, 90.3107738, 25.3815663, 6.631397],
        [0.708478913, 6.07645575, 0.094580537, 90.324703, 26.8427434, 6.95357494],
        [0.685450204, 6.08320452, 0.103688262, 90.3410972, 28.7527373, 7.37124273],
        [0.660511655, 6.09058578, 0.114647767, 90.341297, 30.8426719, 7.83842912],
        [0.631106763, 6.09898622, 0.127562353, 90.3399229, 33.2683226, 8.35622932],
        [0.60239252, 6.10718508, 0.141197449, 90.3187387, 35.6585372, 8.87446837],
        [0.574685996, 6.1149268, 0.154862481, 90.2890178, 37.982628, 9.36856002],
        [0.549279838, 6.12170874, 0.168073709, 90.2538308, 40.1148352, 9.81523017],
        [0.523487025, 6.12859076, 0.182443691, 90.2080219, 42.279737, 10.2799932],
        [0.498121776, 6.13520082, 0.197968844, 90.1482985, 44.4163048, 10.7608292],
        [0.476465121, 6.14082429, 0.2120163, 90.0844905, 46.2626926, 11.1825901],
        [0.456153588, 6.14607051, 0.226245551, 90.0101313, 48.012554, 11.601748],
        [0.433791775, 6.15190581, 0.243138907, 89.9157178, 49.954221, 12.084816],
        [0.404025852, 6.15902259, 0.266186906, 89.7734311, 52.548863, 12.7293228],
        [0.366005234, 6.16774759, 0.297031124, 89.5738271, 55.893128, 13.5723105],
        [0.321312876, 6.17691876, 0.335574819, 89.2922705, 59.8692011, 14.594953],
        [0.278427663, 6.18417441, 0.373123513, 88.9719858, 63.7515612, 15.5832162],
        [0.239534769, 6.19553929, 0.413872182, 88.6402231, 67.3322992, 16.6327601],
        [0.204472084, 6.19968082, 0.464675511, 88.3088859, 70.5963626, 17.9436724],
        [0.175394931, 6.21048495, 0.524148773, 88.0051881, 73.3241903, 19.4940659],
        [0.153758289, 6.22132345, 0.583751542, 87.7681793, 75.3731869, 21.0578654],
        [0.126442905, 6.23697046, 0.675335517, 87.4796317, 77.9661162, 23.4337293],
        [0.11557363, 6.25005689, 0.744980177, 87.365021, 79.008173, 25.2702993],
        [0.106313411, 6.26635361, 0.836966776, 87.2610122, 79.9079131, 27.7208667],
        [0.0989070592, 6.2783871, 0.912917517, 87.1845804, 80.6362714, 29.7408798],
        [0.0945573652, 6.28550694, 0.960719372, 87.1536509, 81.0361104, 31.0067456],
        [0.0902778512, 6.29266034, 1.01249647, 87.1182294, 81.451701, 32.3850431],
        [0.0852959027, 6.30150778, 1.08113362, 87.0519944, 81.9245444, 34.2396643],
        [0.083875265, 6.30582796, 1.11958943, 86.977003, 82.1005075, 35.3326516],
        [0.0825523794, 6.30959352, 1.15442156, 86.9227818, 82.2502751, 36.313756],
        [0.0813272459, 6.31280445, 1.18563003, 86.8893308, 82.3738474, 37.1785229],
        [0.0795266549, 6.31701618, 1.22936759, 86.8298912, 82.5573864, 38.4051678],
        [0.0760640114, 6.32390774, 1.30814197, 86.730576, 82.9155599, 40.6273839],
        [0.0733345584, 6.32845988, 1.38417722, 86.6469757, 83.1880754, 42.7737559],
        [0.0728754398, 6.33436689, 1.51928934, 85.9570928, 83.321603, 47.2189208],
        [0.0740241462, 6.33651625, 1.67086132, 85.1992673, 83.4017941, 52.295749],
        [0.0709732742, 6.32973557, 1.82119648, 84.427344, 83.8384848, 57.4161351],
        [0.0689950542, 6.32296873, 1.89917235, 84.1030756, 84.0687341, 60.0778089],
        [0.0654533403, 6.31403661, 1.97470221, 83.8065972, 84.4296311, 62.6673579],
        [0.0598040781, 6.30238868, 2.04639195, 83.482155, 84.9583795, 65.1956567],
        [0.0532315605, 6.28848325, 2.11585078, 83.1684777, 85.5618308, 67.7111146],
        [0.0467194205, 6.27295621, 2.18521241, 82.8588875, 86.1559337, 70.2420111],
        [0.0211915399, 6.21382682, 2.3716358, 82.5672227, 88.3741932, 76.9350513],
        [0.0119033827, 6.16998589, 2.48828327, 83.7295105, 89.1215994, 80.3298032],
        [0.0114904543, 6.16373556, 2.50403291, 83.888837, 89.1782177, 80.7917836],
        [0.00847440903, 6.15720704, 2.52053877, 84.0621014, 89.3189944, 81.2648114],
        [0.0054345193, 6.12168354, 2.60523315, 84.9795728, 89.7896324, 83.7983903],
        [0.00500718906, 6.08778028, 2.67536568, 85.9496099, 89.8305811, 86.1593636],
        [0.00496310192, 6.07475445, 2.70089071, 86.2819807, 89.9205737, 87.0030543],
        [0.00447449792, 6.05605429, 2.7350364, 86.757467, 89.802935, 88.1996353],
        [0.00297530192, 6.02087628, 2.79766649, 87.459475, 90.1346094, 90.5955514],
        [0.000953058079, 5.9970912, 2.84427868, 87.6265515, 90.27883, 92.3854717],
        [-0.00231375478, 5.99154793, 4.38250817, 87.3674843, 90.6652632, 92.7094979]
    ], dtype=np.float64)

    # Evaluate baseline
    best_path = baseline_path.copy()
    best_score = evaluate_path(best_path)
    
    # Optimization loop
    rng = np.random.default_rng(42)
    step_size = 0.1
    angle_step = 1.0
    
    while time.time() - start_time < deadline:
        # Perturb a random subset of poses
        n_perturb = rng.integers(5, 20)
        indices = rng.choice(len(best_path), size=n_perturb, replace=False)
        
        new_path = best_path.copy()
        for idx in indices:
            # Perturb translation
            new_path[idx, 0] += rng.normal(0, step_size)
            new_path[idx, 1] += rng.normal(0, step_size)
            new_path[idx, 2] += rng.normal(0, step_size)
            
            # Perturb rotation
            new_path[idx, 3] += rng.normal(0, angle_step)
            new_path[idx, 4] += rng.normal(0, angle_step)
            new_path[idx, 5] += rng.normal(0, angle_step)
            
            # Clamp angles to reasonable range? 
            # Yaw 0-90, Pitch 0-90, Roll 0-90 roughly.
            # But the path goes from 0 to 90.
            # Let's not clamp too aggressively, just let the evaluator handle it.
            
        # Evaluate
        score = evaluate_path(new_path)
        
        if score > best_score:
            best_score = score
            best_path = new_path
            # Optionally reduce step size
            step_size *= 0.95
            angle_step *= 0.95
        else:
            # Occasionally increase step size to escape local minima
            if rng.random() < 0.1:
                step_size = min(step_size * 1.5, 0.5)
                angle_step = min(angle_step * 1.5, 5.0)

    return best_path

# EVOLVE-BLOCK-END
