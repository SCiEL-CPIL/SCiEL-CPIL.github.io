import numpy as np
import time

N = 11

def construct():
    start = time.time()
    deadline = start + 140.0
    
    def hex_verts(cx, cy):
        return [(cx + np.cos(i * np.pi / 3), cy + np.sin(i * np.pi / 3)) for i in range(6)]
    
    def min_side(pos_list):
        max_val = 0.0
        for cx, cy in pos_list:
            for vx, vy in hex_verts(cx, cy):
                for i in range(6):
                    a = np.radians(30 + 60 * i)
                    val = vx * np.cos(a) + vy * np.sin(a)
                    if val > max_val:
                        max_val = val
        return max_val / (np.sqrt(3) / 2)
    
    def check_overlap(pos_list):
        for i in range(len(pos_list)):
            for j in range(i + 1, len(pos_list)):
                dx = pos_list[i][0] - pos_list[j][0]
                dy = pos_list[i][1] - pos_list[j][1]
                if dx * dx + dy * dy < 1.99 * 1.99:
                    return False
        return True
    
    configs = []
    
    # Config A: 7 cluster + 4 at distance 4, 30° offset
    pa = [(0.0, 0.0)]
    for i in range(6):
        a = i * np.pi / 3
        pa.append((2.0 * np.cos(a), 2.0 * np.sin(a)))
    for i in range(4):
        a = (i + 0.5) * np.pi / 3
        pa.append((4.0 * np.cos(a), 4.0 * np.sin(a)))
    configs.append(pa)
    
    # Config B: 7 cluster + 4 at distance 3.5, 0° offset (aligned with ring)
    pb = [(0.0, 0.0)]
    for i in range(6):
        a = i * np.pi / 3
        pb.append((2.0 * np.cos(a), 2.0 * np.sin(a)))
    for i in range(4):
        a = i * np.pi / 3
        pb.append((4.0 * np.cos(a), 4.0 * np.sin(a)))
    configs.append(pb)
    
    best_side = 40.0
    best_pos = None
    
    for pos in configs:
        if not check_overlap(pos):
            continue
        side = min_side(pos)
        if side < best_side:
            best_side = side
            best_pos = [list(p) for p in pos]
        
        step = 0.05
        for it in range(5000):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step)]:
                    np_ = [list(p) for p in best_pos]
                    np_[i] = [np_[i][0] + dx, np_[i][1] + dy]
                    if not check_overlap(np_):
                        continue
                    ns = min_side(np_)
                    if ns < best_side - 1e-10:
                        best_side = ns
                        improved = True
            if not improved:
                step *= 0.95
                if step < 1e-4:
                    break
    
    inner = [[best_pos[i][0], best_pos[i][1], 0.0] for i in range(N)]
    return inner, (0.0, 0.0), best_side, 0.0
