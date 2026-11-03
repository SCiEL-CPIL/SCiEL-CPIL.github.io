import numpy as np
import time

N = 12
DEADLINE = 145.0

def hex_vertices(cx, cy, angle_deg, scale=1.0):
    angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3
    r = scale
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def sat_overlap(va, vb):
    """Returns (overlapping, axis, dist). dist<0 means penetration."""
    min_pen = float('inf')
    best_ax = None
    for verts in (va, vb):
        for i in range(6):
            j = (i + 1) % 6
            e = verts[j] - verts[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            pa = np.dot(va, n)
            pb = np.dot(vb, n)
            ma, Ma = pa.min(), pa.max()
            mb, Mb = pb.min(), pb.max()
            if Ma <= mb:
                return (False, n, mb - Ma)
            if Mb <= ma:
                return (False, -n, ma - Mb)
            pen = min(Ma, Mb) - max(ma, mb)
            if pen < min_pen:
                min_pen = pen
                best_ax = n
    if best_ax is None:
        return (False, np.array([1.0, 0.0]), 0.0)
    ca = va.mean(axis=0)
    cb = vb.mean(axis=0)
    if np.dot(cb - ca, best_ax) < 0:
        best_ax = -best_ax
    return (True, best_ax, -min_pen)

def min_outer_side(pos_list, outer_angle=0.0):
    s32 = np.sqrt(3) / 2
    mx = 0.0
    for (x, y), a in pos_list:
        v = hex_vertices(x, y, a)
        for k in range(6):
            th = np.radians(outer_angle) + k * np.pi / 3
            nk = np.array([np.cos(th), np.sin(th)])
            r = np.dot(v, nk).max() / s32
            if r > mx:
                mx = r
    return mx

def is_valid(positions, side):
    """Check all hexagons are non-overlapping and contained."""
    verts_list = [hex_vertices(p[0], p[1], a) for p, a in positions]
    # Check containment
    s32 = np.sqrt(3) / 2
    for v in verts_list:
        for k in range(6):
            th = k * np.pi / 3
            nk = np.array([np.cos(th), np.sin(th)])
            if np.dot(v, nk).max() > side * s32 + 1e-9:
                return False
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            ov, _, d = sat_overlap(verts_list[i], verts_list[j])
            if ov and d < -1e-9:
                return False
    return True

def construct():
    t0 = time.time()
    best = float('inf')
    best_pos = None
    best_ang = None
    
    n_trials = 12
    for tr in range(n_trials):
        if time.time() - t0 > DEADLINE - 8:
            break
        
        rng = np.random.RandomState(tr * 997 + 13)
        
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        idx = 0
        for row in range(3):
            for col in range(4):
                x = col * 2.0
                y = row * np.sqrt(3)
                if row % 2 == 1:
                    x += 1.0
                pos[idx] = [x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3)]
                ang[idx] = rng.uniform(0, 60)
                idx += 1
        
        pos -= pos.mean(axis=0)
        
        vel = np.zeros_like(pos)
        
        dt = 0.005
        fric = 0.88
        rep = 5.0
        cont = 0.1
        
        t_best = float('inf')
        t_bp = pos.copy()
        t_ba = ang.copy()
        
        max_it = 5000
        for it in range(max_it):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            # Pre-compute vertices
            verts = [hex_vertices(pos[i][0], pos[i][1], ang[i]) for i in range(N)]
            
            F = np.zeros_like(pos)
            for i in range(N):
                for j in range(i + 1, N):
                    ov, ax, d = sat_overlap(verts[i], verts[j])
                    if ov:
                        f = rep * max(abs(d), 0.02)
                        F[i] -= f * ax
                        F[j] += f * ax
            
            F -= cont * pos
            
            vel = vel * fric + F * dt
            pos += vel * dt
            ang += 0.01 * rng.standard_normal(N) * (1.0 if it < max_it // 2 else 0.3)
            
            if it % 100 == 0 or it == max_it - 1:
                side = min_outer_side(list(zip(pos[:, 0], pos[:, 1])), 0.0)
                if side < t_best:
                    t_best = side
                    t_bp = pos.copy()
                    t_ba = ang.copy()
            
            if it > max_it // 3:
                rep *= 0.997
                cont *= 1.003
        
        # Try outer angles
        for oa in [0.0, 30.0]:
            s = min_outer_side(list(zip(t_bp[:, 0], t_bp[:, 1])), oa)
            if s < t_best:
                t_best = s
        
        if t_best < best:
            best = t_best
            best_pos = t_bp
            best_ang = t_ba
    
    if best_pos is None:
        pos = np.zeros((N, 2))
        idx = 0
        for row in range(3):
            for col in range(4):
                x = col * 2.0
                y = row * np.sqrt(3)
                if row % 2 == 1:
                    x += 1.0
                pos[idx] = [x, y]
                idx += 1
        pos -= pos.mean(axis=0)
        best_pos = pos
        best_ang = np.zeros(N)
        best = min_outer_side(list(zip(pos[:, 0], pos[:, 1])), 0.0)
    
    inner = [[float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(best), 0.0
