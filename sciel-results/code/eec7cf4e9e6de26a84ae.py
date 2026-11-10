# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580  # safety margin before 630s kill

def _rotation_matrix(roll, pitch, yaw):
    r = np.radians(roll)
    p = np.radians(pitch)
    y = np.radians(yaw)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rx @ Ry @ Rz


def _in_corridor_vec(pts):
    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
    c1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
    c2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
    c3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
    return c1 | c2 | c3


def _sofa_corners(x, y, z, yaw, pitch, roll):
    """Return 8 corners of the 4x4x4 grid-point bounding box (0.2x0.2x0.2 units).
    The grid spacing is 0.05, and 4 points span 3*0.05=0.15... 
    Actually the block is 4x4x4 adjacent grid points. Grid spacing:
    x: linspace(-4,0,81) -> spacing = 4/80 = 0.05
    y: linspace(0,1,21) -> spacing = 1/20 = 0.05
    z: linspace(0,1,21) -> spacing = 1/20 = 0.05
    So 4 adjacent points span 3*0.05 = 0.15 in each dimension.
    The bounding box of the block is 0.15 x 0.15 x 0.15.
    But wait - the volume is 0.05^3 * count. So each grid point occupies 0.05^3.
    A 4x4x4 block of points occupies (4*0.05)^3 = 0.2^3 volume? No...
    Volume = 0.05^3 * number_of_blocks. So each block contributes 0.05^3.
    The block is defined by 4x4x4 adjacent grid points all surviving.
    The bounding box of 4 adjacent points (spacing 0.05) is 0.15 in each dim.
    But the "sofa" that needs to fit is the bounding box of all grid points in the block.
    
    Actually re-reading: "The block is the grid x in linspace(-4,0,81), y and z in linspace(0,1,21)."
    "A grid point survives if it is in the corridor at the first pose and after each move..."
    "Volume = 0.05^3 times the number of 4x4x4 blocks of adjacent grid points that all survive."
    
    So the grid points are the "sofa" - each grid point is a point that must stay in corridor.
    The 4x4x4 block is just a counting unit. The actual constraint is per-grid-point.
    
    So the "sofa" is the set of all 81*21*21 grid points, and each must be in corridor
    at every step. The 4x4x4 block is just how we count volume.
    """
    # The grid points in local frame (before transformation) are:
    # x in linspace(-4,0,81), y in linspace(0,1,21), z in linspace(0,1,21)
    # But wait - the grid is fixed. The transformation R*p + t maps local to world.
    # So the "sofa" in world frame is R*grid + t for each grid point.
    pass
    # Actually I need to think about this differently.
    # The grid points are FIXED in local coordinates. The pose (x,y,z,yaw,pitch,roll)
    # transforms them to world. All grid points must be in the corridor in world frame.
    #
    # Grid: x in [-4,0], y in [0,1], z in [0,1] (local frame)
    # World: corridor is the L-shape described.
    #
    # Start pose: (-3,0,0,0,0,0) -> R=I, so world = local + (-3,0,0)
    #   Grid x in [-4,0] -> world x in [-7,-3]... that's way outside corridor.
    #   Wait, that can't be right. Let me re-read.
    #
    # "The evaluator prepends (-3,0,0,0,0,0)"
    # "The block is the grid x in linspace(-4,0,81), y and z in linspace(0,1,21)."
    #
    # Hmm, with pose (-3,0,0,0,0,0): R=I, T=(-3,0,0)
    # Grid point (gx, gy, gz) -> world (gx-3, gy, gz)
    # gx in [-4,0] -> world x in [-7,-3]
    # That's entirely outside the corridor (which has x>=-4 in the first segment... 
    # actually first segment is x<=0, 0<=y<=1, 0<=z<=1)
    #
    # Wait, x<=0 includes x=-7,-3. And 0<=y<=1, 0<=z<=1.
    # So grid points with gx in [-4,0] map to world x in [-7,-3].
    # For these to be in corridor segment 1: x<=0 (yes), 0<=y<=1 (gy in [0,1], yes), 0<=z<=1 (gz in [0,1], yes).
    # So ALL grid points are in corridor at the start pose! Good.
    #
    # End pose: (0,6,7,90,90,90)
    # R = Rx(90)Ry(90)Rz(90)
    # This rotates the grid significantly. Let me not compute this now.
    pass


def _grid_points():
    gx = np.linspace(-4, 0, 81)
    gy = np.linspace(0, 1, 21)
    gz = np.linspace(0, 1, 21)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
    P = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)
    return P, gx, gy, gz


def _pose_feasible(poses, P):
    """Check if all grid points are in corridor for all poses.
    poses: (M, 6), P: (N, 3)
    Returns bool.
    """
    M = len(poses)
    for m in range(M):
        x, y, z, yaw, pitch, roll = poses[m]
        R = _rotation_matrix(roll, pitch, yaw)
        Pw = P @ R.T + np.array([x, y, z])
        if not _in_corridor_vec(Pw).all():
            return False
    return True


def _densify(poses, pos_step=0.05, ang_step=2.0):
    """Densify poses for the evaluator."""
    if len(poses) == 0:
        return poses
    out = [poses[0].copy()]
    for i in range(len(poses) - 1):
        a = poses[i]; b = poses[i + 1]
        dpos = np.linalg.norm(b[:3] - a[:3])
        dang = float(np.max(np.abs(b[3:] - a[3:])))
        npos = max(1, int(np.ceil(dpos / pos_step - 1e-12)))
        nang = max(1, int(np.ceil(dang / ang_step - 1e-12)))
        n = max(npos, nang)
        for k in range(1, n + 1):
            t = k / n
            out.append(a + t * (b - a))
    out[-1] = poses[-1].copy()
    arr = np.array(out, dtype=np.float64)
    keep = [0]
    for i in range(1, len(arr)):
        if np.any(np.abs(arr[i] - arr[i - 1]) > 1e-12):
            keep.append(i)
    return arr[keep]


def _seed_path():
    return np.array([
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


def _evaluate_blocks(poses, P, gx, gy, gz):
    """Count surviving 4x4x4 blocks."""
    M = len(poses)
    N = P.shape[0]
    nx, ny, nz = len(gx), len(gy), len(gz)
    survives = np.ones(N, dtype=bool)
    
    for m in range(M):
        x, y, z, yaw, pitch, roll = poses[m]
        R = _rotation_matrix(roll, pitch, yaw)
        Pw = P @ R.T + np.array([x, y, z])
        in_cor = _in_corridor_vec(Pw)
        survives &= in_cor
        if not survives.any():
            break
    
    surv_grid = survives.reshape(nx, ny, nz)
    bs = 4
    count = 0
    for i in range(nx - bs + 1):
        for j in range(ny - bs + 1):
            for k in range(nz - bs + 1):
                if surv_grid[i:i+bs, j:j+bs, k:k+bs].all():
                    count += 1
    return count


def _local_optimize(poses, P, gx, gy, gz, max_iters=50, timeout=400):
    """Simple local optimization: perturb waypoints and keep if score improves."""
    start = time.time()
    # Densify for evaluation
    dens = _densify(poses)
    best_score = _evaluate_blocks(dens, P, gx, gy, gz)
    best_poses = poses.copy()
    
    rng = np.random.default_rng(42)
    n = len(poses)
    
    for it in range(max_iters):
        if time.time() > start + timeout:
            break
        # Pick a random waypoint to perturb
        idx = rng.integers(1, n - 1)
        orig = best_poses[idx].copy()
        
        # Perturb position and angles
        pert = best_poses[idx].copy()
        pert[0] += rng.normal(0, 0.02)
        pert[1] += rng.normal(0, 0.02)
        pert[2] += rng.normal(0, 0.02)
        pert[3] += rng.normal(0, 1.0)
        pert[4] += rng.normal(0, 1.0)
        pert[5] += rng.normal(0, 1.0)
        
        test_poses = best_poses.copy()
        test_poses[idx] = pert
        test_dens = _densify(test_poses)
        score = _evaluate_blocks(test_dens, P, gx, gy, gz)
        
        if score > best_score:
            best_score = score
            best_poses = test_poses
    
    return best_poses, best_score


def moving_sofa_3d_path() -> np.ndarray:
    """
    Strategy: Use the seed path as a baseline, then apply targeted local
    optimization to improve the number of surviving 4x4x4 blocks.
    The key insight is that the seed path already works (score 1.0), so we
    start from a known-feasible solution and optimize locally.
    """
    P, gx, gy, gz = _grid_points()
    
    # Start with seed path
    seed = _seed_path()
    dens = _densify(seed)
    base_score = _evaluate_blocks(dens, P, gx, gy, gz)
    
    # Local optimization with time budget
    remaining = _DEADLINE - time.time() - 30  # 30s safety margin
    if remaining > 10:
        optimized, opt_score = _local_optimize(seed, P, gx, gy, gz, 
                                                  max_iters=200, timeout=min(remaining, 500))
        if opt_score > base_score:
            seed = optimized
            base_score = opt_score
    
    # Final densification
    result = _densify(seed)
    return result


# EVOLVE-BLOCK-END
