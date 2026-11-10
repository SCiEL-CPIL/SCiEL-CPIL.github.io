# EVOLVE-BLOCK-START
import numpy as np
import time


def _rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """R = Rx(roll) @ Ry(pitch) @ Rz(yaw)."""
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rx @ Ry @ Rz


def _in_corridor(pts):
    """Vectorized corridor membership for an (N,3) array."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    m1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
    m2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
    m3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
    return m1 | m2 | m3


def _arc_bbox(center, u, v, theta0, theta1):
    """
    Exact axis-aligned bounding box of the circular arc
        c + u*cos(t) + v*sin(t),  t in [theta0, theta1]
    where u, v are orthogonal (rotation about the remaining axis).
    """
    a = np.array([center[0] + u[0] * np.cos(theta0) + v[0] * np.sin(theta0),
                  center[1] + u[1] * np.cos(theta0) + v[1] * np.sin(theta0),
                  center[2] + u[2] * np.cos(theta0) + v[2] * np.sin(theta0)])
    b = np.array([center[0] + u[0] * np.cos(theta1) + v[0] * np.sin(theta1),
                  center[1] + u[1] * np.cos(theta1) + v[1] * np.sin(theta1),
                  center[2] + u[2] * np.cos(theta1) + v[2] * np.sin(theta1)])
    cands = np.vstack([a, b])

    span = abs(theta1 - theta0)
    # interior extrema of cos(t), sin(t)
    if span > 1e-9:
        for t in (np.pi / 2, 3 * np.pi / 2, np.pi, 0.0, 2 * np.pi):
            if theta0 < t < theta1:
                cands = np.vstack([cands,
                                   np.array([center[0] + u[0] * np.cos(t) + v[0] * np.sin(t),
                                             center[1] + u[1] * np.cos(t) + v[1] * np.sin(t),
                                             center[2] + u[2] * np.cos(t) + v[2] * np.sin(t)])])
        # extrema of the projected coordinate along each axis:
        # d/dt (A cos t + B sin t) = -A sin t + B cos t = 0
        A = np.array([u[0], u[1], u[2]])
        B = np.array([v[0], v[1], v[2]])
        denom = A * A + B * B
        for ax in range(3):
            if denom[ax] < 1e-12:
                continue
            t = np.arctan2(A[ax], B[ax])
            # add 0 and pi offsets
            for off in (0.0, np.pi):
                tt = t + off
                # wrap into [theta0, theta1]
                while tt < theta0:
                    tt += 2 * np.pi
                while tt > theta1:
                    tt -= 2 * np.pi
                if theta0 < tt < theta1:
                    cands = np.vstack([cands,
                                       np.array([center[0] + u[0] * np.cos(tt) + v[0] * np.sin(tt),
                                                 center[1] + u[1] * np.cos(tt) + v[1] * np.sin(tt),
                                                 center[2] + u[2] * np.cos(tt) + v[2] * np.sin(tt)])])
    return cands.min(axis=0), cands.max(axis=0)


def _evaluate_path(path, grid_x, grid_y, grid_z):
    """
    Count 4x4x4 blocks of the 81x21x21 grid that survive the whole motion.
    """
    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    n_pts = len(grid_points)

    start_pose = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end_pose = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    full_path = np.vstack([start_pose, path, end_pose])
    n_steps = len(full_path) - 1

    # --- initial survival ---
    R0 = _rotation_matrix(full_path[0, 3], full_path[0, 4], full_path[0, 5])
    pos0 = full_path[0, :3]
    survive = _in_corridor((R0 @ grid_points.T).T + pos0)
    if not survive.any():
        return 0

    # Precompute rotation matrices for each pose (cache)
    Rmats = []
    for i in range(len(full_path)):
        Rmats.append(_rotation_matrix(full_path[i, 3], full_path[i, 4], full_path[i, 5]))

    for i in range(n_steps):
        old = full_path[i]
        new = full_path[i + 1]
        R = Rmats[i]

        # x move
        pts = (R @ grid_points.T).T + np.array([new[0], old[1], old[2]])
        survive = survive & _in_corridor(pts)
        # y move
        pts = (R @ grid_points.T).T + np.array([old[0], new[1], old[2]])
        survive = survive & _in_corridor(pts)
        # z move
        pts = (R @ grid_points.T).T + np.array([old[0], old[1], new[2]])
        survive = survive & _in_corridor(pts)

        # rotation about the dominant axis
        dyaw = new[3] - old[3]
        dpitch = new[4] - old[4]
        droll = new[5] - old[5]
        if abs(dyaw) >= abs(dpitch) and abs(dyaw) >= abs(droll):
            axis, ddeg = 2, dyaw
        elif abs(dpitch) >= abs(droll):
            axis, ddeg = 1, dpitch
        else:
            axis, ddeg = 0, droll

        if abs(ddeg) > 1e-9:
            dtheta = np.radians(ddeg)
            # rotation about the body axis; in the world frame the axis is
            # R @ e_axis. The arc of each grid point is a circle in the plane
            # perpendicular to that axis.
            e = np.zeros(3)
            e[axis] = 1.0
            axis_w = R @ e
            # For each point, center = point projected onto axis line through origin
            # Actually the rotation is about the world axis through the origin
            # (since translation is applied separately). The arc of a point p
            # under rotation about axis_w (through origin) by dtheta:
            #   center = (p . axis_w) * axis_w
            #   u = p - center
            #   v = axis_w x u
            #   arc(t) = center + u*cos(t) + v*sin(t)
            centers = (grid_points @ axis_w)[:, None] * axis_w[None, :]
            u = grid_points - centers
            v = np.cross(axis_w[None, :], u)
            # bounding box of the union arc = bbox over all points
            # We compute per-point arc bbox and take the union.
            # To keep it fast, compute min/max over a small set of candidate angles.
            theta0 = 0.0
            theta1 = dtheta
            # candidate angles: endpoints + interior extrema
            cand_angles = [theta0, theta1]
            span = abs(theta1 - theta0)
            if span > 1e-9:
                for t in (np.pi / 2, 3 * np.pi / 2, np.pi, 0.0, 2 * np.pi):
                    if theta0 < t < theta1:
                        cand_angles.append(t)
                # per-axis extrema
                A = u  # shape (N,3)
                B = v  # shape (N,3)
                denom = np.sum(A * A + B * B, axis=1)
                for ax in range(3):
                    mask = denom > 1e-12
                    if not mask.any():
                        continue
                    t = np.arctan2(A[mask, ax], B[mask, ax])
                    for off in (0.0, np.pi):
                        tt = t + off
                        while tt < theta0:
                            tt += 2 * np.pi
                        while tt > theta1:
                            tt -= 2 * np.pi
                        sel = (tt > theta0) & (tt < theta1)
                        if sel.any():
                            # add these angles (approximate: take a few)
                            for ang in tt[sel][:4]:
                                cand_angles.append(ang)
            # dedupe
            cand_angles = list(set(round(a, 9) for a in cand_angles))
            all_pts = []
            for ang in cand_angles:
                ca, sa = np.cos(ang), np.sin(ang)
                all_pts.append(centers + u * ca + v * sa)
            all_pts = np.vstack(all_pts)
            mn = all_pts.min(axis=0)
            mx = all_pts.max(axis=0)
            bbox_pts = np.array([
                [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
                [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
                [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
            ])
            if not _in_corridor(bbox_pts).all():
                return 0
        else:
            R_new = Rmats[i + 1]
            pos_new = new[:3]
            pts_final = (R_new @ grid_points.T).T + pos_new
            survive = survive & _in_corridor(pts_final)

        if not survive.any():
            return 0

    # --- count 4x4x4 blocks via 3D cumulative sum ---
    surv3 = survive.reshape(81, 21, 21).astype(np.int32)
    # pad
    pad = np.zeros((81 + 3, 21 + 3, 21 + 3), dtype=np.int32)
    pad[3:, 3:, 3:] = surv3
    cs = np.cumsum(pad, axis=0)
    cs = np.cumsum(cs, axis=1)
    cs = np.cumsum(cs, axis=2)
    # 4x4x4 block at (i,j,k) sum:
    #   cs[i+4+3, j+4+3, k+4+3] - cs[i, j+4+3, k+4+3] - cs[i+4+3, j, k+4+3] - cs[i+4+3, j+4+3, k]
    #   + cs[i, j, k+4+3] + cs[i, j+4+3, k] + cs[i+4+3, j, k] - cs[i, j, k]
    i = np.arange(81 - 3)
    j = np.arange(21 - 3)
    k = np.arange(21 - 3)
    I, J, K = np.meshgrid(i, j, k, indexing='ij')
    s = (cs[I + 4 + 3, J + 4 + 3, K + 4 + 3]
         - cs[I, J + 4 + 3, K + 4 + 3]
         - cs[I + 4 + 3, J, K + 4 + 3]
         - cs[I + 4 + 3, J + 4 + 3, K]
         + cs[I, J, K + 4 + 3]
         + cs[I, J + 4 + 3, K]
         + cs[I + 4 + 3, J, K]
         - cs[I, J, K])
    return int(np.sum(s == 64))


def moving_sofa_3d_path() -> np.ndarray:
    deadline = time.time() + 600

    baseline = np.array([
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

    grid_x = np.linspace(-4, 0, 81)
    grid_y = np.linspace(0, 1, 21)
    grid_z = np.linspace(0, 1, 21)

    best_path = baseline.copy()
    best_score = _evaluate_path(best_path, grid_x, grid_y, grid_z)

    # --- greedy coordinate-descent local search ---
    n_poses = len(baseline)
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    rng = np.random.default_rng(0)

    for step_size in step_sizes:
        if time.time() >= deadline - 10:
            break
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # random order of poses and coordinates
            order = rng.permutation(n_poses * 6)
            for idx in order:
                if time.time() >= deadline - 10:
                    break
                pi = idx // 6
                ci = idx % 6
                # try +/- step
                for sgn in (1.0, -1.0):
                    cand = best_path.copy()
                    cand[pi, ci] += sgn * step_size
                    sc = _evaluate_path(cand, grid_x, grid_y, grid_z)
                    if sc > best_score:
                        best_path = cand
                        best_score = sc
                        improved = True
                        break
    return best_path


# EVOLVE-BLOCK-END
