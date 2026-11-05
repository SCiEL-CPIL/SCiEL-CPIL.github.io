import math
import random
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Helper: Support function of unit hexagon
    def h_unit(theta):
        delta = theta % (math.pi / 3)
        if delta > math.pi / 6:
            delta = math.pi / 3 - delta
        return math.cos(delta)

    # Helper: Check collision between two hexes
    def check_collision(p1, p2):
        x1, y1, a1 = p1
        x2, y2, a2 = p2
        
        dx = x2 - x1
        dy = y2 - y1
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.0: # Max diameter 2, so centers > 2 apart means disjoint
            return False
        
        # Generate vertices
        verts1 = [(x1 + math.cos(a1 + k * math.pi / 3), y1 + math.sin(a1 + k * math.pi / 3)) for k in range(6)]
        verts2 = [(x2 + math.cos(a2 + k * math.pi / 3), y2 + math.sin(a2 + k * math.pi / 3)) for k in range(6)]
        
        # Check separating axes
        for k in range(6):
            ang = a1 + k * math.pi / 3 + math.pi / 6
            nx = math.cos(ang)
            ny = math.sin(ang)
            
            min1 = float('inf'); max1 = float('-inf')
            min2 = float('inf'); max2 = float('-inf')
            
            for v in verts1:
                proj = v[0]*nx + v[1]*ny
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
                
            for v in verts2:
                proj = v[0]*nx + v[1]*ny
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
                
            if max1 < min2 or max2 < min1:
                return False
                
        for k in range(6):
            ang = a2 + k * math.pi / 3 + math.pi / 6
            nx = math.cos(ang)
            ny = math.sin(ang)
            
            min1 = float('inf'); max1 = float('-inf')
            min2 = float('inf'); max2 = float('-inf')
            
            for v in verts1:
                proj = v[0]*nx + v[1]*ny
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
                
            for v in verts2:
                proj = v[0]*nx + v[1]*ny
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
                
            if max1 < min2 or max2 < min1:
                return False
                
        return True

    def is_valid(positions):
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if check_collision(positions[i], positions[j]):
                    return False
        return True

    def calc_side_length(positions, phi):
        max_proj = 0.0
        for j in range(6):
            ang_n = phi + j * math.pi / 3
            nx = math.cos(ang_n)
            ny = math.sin(ang_n)
            
            for p in positions:
                x, y, a = p
                proj_center = x*nx + y*ny
                h_val = h_unit(ang_n - a)
                total_proj = proj_center + h_val
                if total_proj > max_proj:
                    max_proj = total_proj
                    
        return max_proj / (math.sqrt(3)/2)

    best_score = float('inf')
    best_pos = None
    best_phi = 0.0

    for restart in range(20): # Max 20 restarts
        if time.time() > deadline - 5.0:
            break
            
        # Initialize
        pos = []
        pos.append([0.0, 0.0, 0.0])
        for k in range(6):
            angle = k * math.pi / 3
            x = math.sqrt(3) * math.cos(angle)
            y = math.sqrt(3) * math.sin(angle)
            pos.append([x, y, 0.0])
        for k in range(4):
            angle = k * math.pi / 3 + math.pi/6
            x = 2 * math.sqrt(3) * math.cos(angle)
            y = 2 * math.sqrt(3) * math.sin(angle)
            pos.append([x, y, 0.0])
            
        # Add some noise
        if restart > 0:
            for p in pos:
                p[0] += random.uniform(-0.5, 0.5)
                p[1] += random.uniform(-0.5, 0.5)
                p[2] += random.uniform(-math.pi/6, math.pi/6)
                
        phi = random.uniform(0, math.pi/3)
        
        if not is_valid(pos):
            continue
            
        current_score = calc_side_length(pos, phi)
        if current_score < best_score:
            best_score = current_score
            best_pos = [p[:] for p in pos]
            best_phi = phi
            
        # Local Search
        step_size = 0.1
        no_improve = 0
        
        while time.time() < deadline - 2.0:
            idx = random.randint(0, N-1)
            var_type = random.randint(0, 3)
            
            old_val = 0
            if var_type == 0:
                old_val = pos[idx][0]
                pos[idx][0] += random.uniform(-step_size, step_size)
            elif var_type == 1:
                old_val = pos[idx][1]
                pos[idx][1] += random.uniform(-step_size, step_size)
            elif var_type == 2:
                old_val = pos[idx][2]
                pos[idx][2] += random.uniform(-step_size, step_size)
            else:
                old_val = phi
                phi += random.uniform(-step_size, step_size)
                
            new_score = calc_side_length(pos, phi)
            
            # Check validity
            valid = True
            # Only check collisions involving the moved hexagon
            if var_type < 3:
                for j in range(N):
                    if j != idx:
                        if check_collision(pos[idx], pos[j]):
                            valid = False
                            break
            else:
                valid = is_valid(pos) # Phi change doesn't affect collisions
                
            if valid and new_score < best_score:
                best_score = new_score
                best_pos = [p[:] for p in pos]
                best_phi = phi
                no_improve = 0
                step_size = max(0.01, step_size * 0.99)
            else:
                # Revert
                if var_type == 0:
                    pos[idx][0] = old_val
                elif var_type == 1:
                    pos[idx][1] = old_val
                elif var_type == 2:
                    pos[idx][2] = old_val
                else:
                    phi = old_val
                no_improve += 1
                
            if no_improve > 2000:
                break

    if best_pos is None:
        # Fallback to initial honeycomb
        pos = []
        pos.append([0.0, 0.0, 0.0])
        for k in range(6):
            angle = k * math.pi / 3
            x = math.sqrt(3) * math.cos(angle)
            y = math.sqrt(3) * math.sin(angle)
            pos.append([x, y, 0.0])
        for k in range(4):
            angle = k * math.pi / 3 + math.pi/6
            x = 2 * math.sqrt(3) * math.cos(angle)
            y = 2 * math.sqrt(3) * math.sin(angle)
            pos.append([x, y, 0.0])
        best_pos = pos
        best_score = calc_side_length(pos, 0.0)
        best_phi = 0.0

    inner = []
    for p in best_pos:
        inner.append([p[0], p[1], p[2] * 180.0 / math.pi])
        
    return inner, (0.0, 0.0), best_score, best_phi * 180.0 / math.pi
# EVOLVE-BLOCK-END
