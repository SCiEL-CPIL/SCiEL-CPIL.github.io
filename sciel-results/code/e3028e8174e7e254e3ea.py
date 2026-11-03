"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 12
DEADLINE = 140.0

def construct():
    t0 = time.time()
    
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        a = math.radians(angle_deg)
        verts = np.array([[cx + side * math.cos(a + i * math.pi / 3),
                          cy + side * math.sin(a + i * math.pi / 3)] for i in range(6)])
        return verts
    
    def sat_overlap(verts1, verts2, tol=1e-7):
        edges = []
        for v in [verts1, verts2]:
            for i in range(6):
                edge = v[(i+1) % 6] - v[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    edges.append(normal / norm)
        for normal in edges:
            proj1 = verts1 @ normal
            proj2 = verts2 @ normal
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return False
        return True
    
    def min_translation(verts1, verts2):
        edges = []
        for v in [verts1, verts2]:
            for i in range(6):
                edge = v[(i+1) % 6] - v[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    edges.append(normal / norm)
        min_depth = np.inf
        min_normal = None
        for normal in edges:
            proj1 = verts1 @ normal
            proj2 = verts2 @ normal
            depth = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
            if depth > 0 and depth < min_depth:
                min_depth = depth
                min_normal = normal
        if min_normal is None:
            return 0.0, np.array([0.0, 0.0])
        c1, c2 = verts1.mean(axis=0), verts2.mean(axis=0)
        if np.dot(c1 - c2, min_normal) < 0:
            min_normal = -min_normal
        return min_depth, min_normal
    
    def required_side(positions, angles):
        max_s = 0.0
        for i in range(N):
            v = hex_vertices(positions[i,0], positions[i,1], angles[i])
            for px, py in v:
                for k in range(6):
                    theta = math.pi/6 + k * math.pi/3
                    proj = px * math.cos(theta) + py * math.sin(theta)
                    s = proj / (math.sqrt(3)/2)
                    if s > max_s:
                        max_s = s
        return max(0.0, max_s)
    
    def resolve_collisions(positions, angles, iters=30):
        for _ in range(iters):
            verts = [hex_vertices(positions[i,0], positions[i,1], angles[i]) for i in range(N)]
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    depth, normal = min_translation(verts[i], verts[j])
                    if depth > 1e-5:
                        positions[i] -= 0.5 * depth * normal
                        positions[j] += 0.5 * depth * normal
                        moved = True
            if not moved:
                break
    
    # Initialize: honeycomb cluster
    d = math.sqrt(3)
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    # 3 rows x 4 cols with hex offset
    for i in range(N):
        row, col = divmod(i, 4)
        positions[i, 0] = col * d * math.sqrt(3) + (d * math.sqrt(3)/2 if row % 2 else 0)
        positions[i, 1] = row * d * 1.5
        angles[i] = 0.0
    positions -= positions.mean(axis=0)
    
    current_side = required_side(positions, angles)
    best_side = current_side
    best_pos = positions.copy()
    best_ang = angles.copy()
    
    np.random.seed(42)
    T = 1.0
    cooling = 0.998
    min_T = 0.0005
    
    while T > min_T:
        if time.time() - t0 > DEADLINE:
            break
        
        i = np.random.randint(N)
        old_p = positions[i].copy()
        old_a = angles[i]
        
        positions[i] += np.random.normal(0, 0.2, 2)
        angles[i] = (old_a + np.random.normal(0, 0.05)) % 360
        
        # Check overlaps
        vi = hex_vertices(positions[i,0], positions[i,1], angles[i])
        overlap = False
        for j in range(N):
            if j != i:
                vj = hex_vertices(positions[j,0], positions[j,1], angles[j])
                if sat_overlap(vi, vj):
                    overlap = True
                    break
        
        new_side = required_side(positions, angles)
        if overlap:
            delta = 5.0
        else:
            delta = new_side - current_side
        
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            current_side = new_side
            if current_side < best_side:
                best_side = current_side
                best_pos = positions.copy()
                best_ang = angles.copy()
        else:
            positions[i] = old_p
            angles[i] = old_a
        
        T *= cooling
        
        # Periodic resolution
        if int((1.0 - T) / (1.0 - min_T)) % 200 == 0 and T > min_T * 2:
            resolve_collisions(positions, angles, 10)
            current_side = required_side(positions, angles)
            if current_side < best_side:
                best_side = current_side
                best_pos = positions.copy()
                best_ang = angles.copy()
    
    # Final polish
    positions = best_pos.copy()
    angles = best_ang.copy()
    resolve_collisions(positions, angles, 50)
    final_side = required_side(positions, angles)
    
    inner = [[float(positions[i,0]), float(positions[i,1]), float(angles[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(final_side), 0.0

