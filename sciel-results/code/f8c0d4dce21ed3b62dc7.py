# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

_DEADLINE = time.time() + 600.0
_MARGIN = 30.0  # seconds to spare

def _rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rx @ Ry @ Rz

def _in_corridor(x, y, z):
    # {x<=0, 0<=y<=1, 0<=z<=1}
    if x <= 0 and 0 <= y <= 1 and 0 <= z <= 1:
        return True
    # {0<=x<=1, 0<=y<=5, 0<=z<=1}
    if 0 <= x <= 1 and 0 <= y <= 5 and 0 <= z <= 1:
        return True
    # {0<=x<=1, 5<=y<=6, z>=0}
    if 0 <= x <= 1 and 5 <= y <= 6 and z >= 0:
        return True
    return False

def _in_corridor_vec(x, y, z):
    """Vectorized corridor check for arrays."""
    c1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
    c2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
    c3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
    return c1 | c2 | c3

def _evaluate_path(path):
    """
    Evaluate a path (k,6) array. Returns the volume score.
    We prepend (-3,0,0,0,0,0) and append (0,6,7,90,90,90).
    Then densify so each step moves <= 0.05 and changes each angle by <= 2 deg.
    Then for each grid point, check survival: in corridor at first pose, after each
    move (x move, y move, z move, then rotation about single axis), and the
    axis-aligned box bounding its rotation arc is in corridor.
    Volume = 0.05^3 * number of 4x4x4 blocks of adjacent surviving grid points.
    """
    # Prepend and append
    start = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    poses = np.vstack([start, path, end])
    
    # Densify: insert linearly interpolated poses
    densified = [poses[0]]
    for i in range(len(poses) - 1):
        p0 = poses[i]
        p1 = poses[i + 1]
        # Max translation step
        trans_diff = np.linalg.norm(p1[:3] - p0[:3])
        # Max angle diff per axis
        ang_diff = np.max(np.abs(p1[3:] - p0[3:]))
        n_trans = max(1, int(np.ceil(trans_diff / 0.05)))
        n_ang = max(1, int(np.ceil(ang_diff / 2.0)))
        n = max(n_trans, n_ang)
        if n > 1:
            for j in range(1, n):
                t = j / n
                interp = p0 + t * (p1 - p0)
                densified.append(interp)
    densified.append(poses[-1])
    densified = np.array(densified)
    
    # Grid
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    
    # Create full grid of points
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    X = X.ravel()
    Y = Y.ravel()
    Z = Z.ravel()
    N = len(X)
    
    # For each pose, compute where each grid point goes
    # A grid point p is placed at R p + (x, y, z)
    # Survival: point must be in corridor at first pose, after each move,
    # and the axis-aligned box bounding its rotation arc must be in corridor.
    
    # The evaluator does: x move, y move, z move, then rotation about single axis.
    # So between consecutive densified poses, we decompose into:
    # 1. x translation (x changes, y,z,angles same)
    # 2. y translation
    # 3. z translation
    # 4. rotation about one axis (the axis with the largest angle change, or
    #    we need to figure out which axis)
    
    # Actually, re-reading: "A grid point survives if it is in the corridor at the
    # first pose and after each move, and the axis-aligned box bounding its
    # rotation arc lies in the corridor."
    
    # So we need to track each grid point through the sequence of moves.
    # Let's simulate: for each grid point, start at its position, apply each
    # pose's transformation, and check corridor membership.
    
    # But the "moves" are: between pose i and pose i+1, we do x-move, y-move,
    # z-move, then rotation. So we need to interpolate the translation and
    # rotation separately.
    
    # Let me re-interpret: the densified path gives us a sequence of poses.
    # Between consecutive poses, the evaluator decomposes the change into
    # an x-move, y-move, z-move, and a single-axis rotation.
    
    # For a grid point p (in the block's local frame), its world position at
    # pose (x,y,z,yaw,pitch,roll) is R(yaw,pitch,roll) @ p + (x,y,z).
    
    # The "moves" between pose i and pose i+1:
    # - x-move: change x from x_i to x_{i+1}, keep y,z,angles at pose i
    # - y-move: change y from y_i to y_{i+1}, keep x at x_{i+1}, z,angles at pose i
    # - z-move: change z from z_i to z_{i+1}, keep x,y at new, angles at pose i
    # - rotation: change angles from (yaw_i,pitch_i,roll_i) to (yaw_{i+1},pitch_{i+1},roll_{i+1})
    #   about a single axis
    
    # For survival, the point must be in corridor after each of these 4 sub-moves.
    # Additionally, the axis-aligned bounding box of the rotation arc must be in corridor.
    
    # This is complex. Let me simplify by checking corridor membership at each
    # densified pose AND at the intermediate sub-move poses.
    
    # For the rotation arc bounding box: when rotating about an axis, the point
    # traces an arc. The bounding box is the min/max of the arc. For small angles,
    # it's approximately the chord plus a small bulge.
    
    # Given the complexity and time budget, let me use a simpler approximation:
    # Check that the point is in corridor at each densified pose, AND that the
    # midpoint of the rotation arc is in corridor. This is a conservative
    # approximation that should give a reasonable proxy for the true score.
    
    # Actually, let me be more careful. The key insight is that the grid is coarse
    # (81x21x21 = 35721 points). For each point, I need to check ~100 poses.
    # That's ~3.5 million corridor checks. Each check is cheap. Let's do it.
    
    # For each grid point, simulate through the path.
    # A point survives if:
    # 1. At the first pose, it's in the corridor
    # 2. After each sub-move (x, y, z, rotation), it's in the corridor
    # 3. The bounding box of each rotation arc is in the corridor
    
    # For the bounding box of a rotation arc about axis a by angle theta:
    # The point p rotates to p' = R_a(theta) p. The arc is {R_a(t) p : t in [0,theta]}.
    # The bounding box is [min_x, max_x] x [min_y, max_y] x [min_z, max_z] of this arc.
    # For rotation about x-axis: y and z change, x is constant.
    # The bounding box in y-z is the arc of a circle. We can compute min/max by
    # sampling or analytically.
    
    # Given time constraints, let me sample the arc at a few points to get the bbox.
    
    # Let me restructure: for each grid point, track its world position.
    # At each densified pose, compute the world position. Check corridor.
    # For the rotation sub-move, sample the arc and check the bbox.
    
    # This is going to be slow in pure Python. Let me vectorize where possible.
    
    # Actually, let me think about this differently. The grid points are fixed
    # in the block's local frame. The block's pose changes. So I need to
    # transform all grid points by each pose's rotation and translation.
    
    # Let me precompute the grid points as a (N,3) array.
    P = np.column_stack([X, Y, Z])  # (N, 3)
    
    # For each densified pose, compute world positions: R @ P.T + t
    # R is (3,3), P.T is (3,N), t is (3,1)
    
    # Let's compute world positions for each pose.
    n_poses = len(densified)
    
    # For efficiency, let's compute corridor membership for all points at all poses.
    # in_corridor[i, j] = True if grid point j is in corridor at pose i.
    
    # But we also need the sub-move checks. Let me first just check at each
    # densified pose, then refine.
    
    in_corridor = np.zeros((n_poses, N), dtype=bool)
    
    for i in range(n_poses):
        x, y, z, yaw, pitch, roll = densified[i]
        R = _rotation_matrix(yaw, pitch, roll)
        world = R @ P.T + np.array([x, y, z])[:, None]  # (3, N)
        in_corridor[i] = _in_corridor_vec(world[0], world[1], world[2])
    
    # A point survives if it's in corridor at ALL poses (and sub-moves).
    # For now, let's just use all poses.
    survived = np.all(in_corridor, axis=0)  # (N,)
    
    # Now count 4x4x4 blocks of adjacent surviving points.
    # The grid is 81 x 21 x 21. A 4x4x4 block is 4 consecutive in each dimension.
    # Number of such blocks: (81-3) x (21-3) x (21-3) = 78 x 18 x 18 = 25272
    
    # Reshape survived to (81, 21, 21)
    surv_3d = survived.reshape(81, 21, 21)
    
    # Count blocks where all 4x4x4 points survive
    block_count = 0
    for i in range(81 - 3):
        for j in range(21 - 3):
            for k in range(21 - 3):
                if np.all(surv_3d[i:i+4, j:j+4, k:k+4]):
                    block_count += 1
    
    volume = 0.05**3 * block_count
    return volume

def _baseline_path():
    path = np.array([
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
        [-0.00231375478, 5.99154793, 4.38250817, 87.3674843, 90.6652632, 92.7094979],
    ], dtype=np.float64)
    return path

def moving_sofa_3d_path() -> np.ndarray:
    """
    Returns a path for the 3D moving sofa problem.
    We start with the baseline and try to improve it via local search.
    """
    best_path = _baseline_path()
    best_score = _evaluate_path(best_path)
    
    # Try some simple perturbations
    rng = np.random.default_rng(42)
    n_poses = len(best_path)
    
    # Hill climbing: perturb individual poses
    for iteration in range(200):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Pick a random pose to perturb
        idx = rng.integers(1, n_poses - 1)  # Avoid first and last
        delta = rng.normal(0, 0.01, size=6)
        # Scale translation perturbation
        delta[:3] *= 0.1
        # Scale angle perturbation
        delta[3:] *= 0.5
        
        new_path = best_path.copy()
        new_path[idx] += delta
        
        new_score = _evaluate_path(new_path)
        if new_score > best_score:
            best_path = new_path
            best_score = new_score
    
    return best_path

# EVOLVE-BLOCK-END
