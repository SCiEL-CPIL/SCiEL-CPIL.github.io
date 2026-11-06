import random
import math
import time
from collections import defaultdict

def get_ag24_lines():
    """
    Returns the 20 lines of AG(2,4) as sets of 4 indices (0..15).
    Indices map to (i, j) in GF(4)^2.
    GF(4) elements: 0, 1, 2, 3 corresponding to 0, 1, alpha, alpha+1.
    Addition table for +1 (shift):
    0->1, 1->0, 2->3, 3->2
    Multiplication table? We need lines y = mx + c.
    Slopes: 0, 1, 2, 3, inf.
    """
    # Map GF(4) to 0,1,2,3
    # 0:0, 1:1, 2:alpha, 3:alpha+1
    # Add 1: 0->1, 1->0, 2->3, 3->2
    add1 = {0: 1, 1: 0, 2: 3, 3: 2}
    
    # Mult by 2 (alpha):
    # 0*2=0
    # 1*2=2
    # 2*2=2*2=4=2+1=3 (since alpha^2 = alpha+1)
    # 3*2=(2+1)*2 = 4+2 = 3+2 = 1 (since 3=alpha+1, 3*2 = alpha^2+2 = 3+2 = 1)
    mult2 = {0: 0, 1: 2, 2: 3, 3: 1}
    
    # Mult by 3 (alpha+1):
    # 0*3=0
    # 1*3=3
    # 2*3=2*(2+1)=6+2=3+2=1 (Wait. 2*3 = alpha(alpha+1) = alpha^2+alpha = (alpha+1)+alpha = 1)
    # 3*3=(alpha+1)^2 = alpha^2+2alpha+1 = (alpha+1)+2alpha+1 = 3alpha = 3(alpha) = 3*2 = 1? No.
    # 3*3 = (alpha+1)(alpha+1) = alpha^2 + 2alpha + 1 = (alpha+1) + 0 + 1 = alpha = 2.
    mult3 = {0: 0, 1: 3, 2: 1, 3: 2}
    
    lines = []
    
    # Slope infinity (vertical lines x=c)
    for c in range(4):
        line = []
        for r in range(4):
            line.append(c * 4 + r) # x=c, y=r. Index = x*4+y
        lines.append(set(line))
        
    # Slope 0 (horizontal lines y=c)
    for c in range(4):
        line = []
        for r in range(4):
            line.append(r * 4 + c) # x=r, y=c
        lines.append(set(line))
        
    # Slope 1 (y = x + c)
    for c in range(4):
        line = []
        for x in range(4):
            y = add1.get(x if c==1 else x) # This logic is messy.
            # y = x + c
            # If c=0: y=x
            # If c=1: y=x+1
            # If c=2: y=x+2 = x+alpha
            # If c=3: y=x+3 = x+alpha+1
            
            # Let's just compute x+c in GF(4)
            # We need addition in GF(4).
            pass
            
    # Let's define addition properly
    def add_gf4(a, b):
        # 0:0, 1:1, 2:2, 3:3
        # XOR if we use bits?
        # 0=00, 1=01, 2=10, 3=11
        return a ^ b
        
    def mul_gf4(a, b):
        if a == 0 or b == 0: return 0
        if a == 1: return b
        if b == 1: return a
        # 2*2=3, 2*3=1, 3*3=2
        table = {
            (2,2): 3, (2,3): 1, (3,2): 1, (3,3): 2
        }
        return table.get((a,b), 0)
        
    lines = []
    # Vertical
    for c in range(4):
        line = set()
        for r in range(4):
            line.add(c * 4 + r)
        lines.append(line)
        
    # Horizontal
    for c in range(4):
        line = set()
        for r in range(4):
            line.add(r * 4 + c)
        lines.append(line)
        
    # Slope 1: y = x + c
    for c in range(4):
        line = set()
        for x in range(4):
            y = add_gf4(x, c)
            line.add(x * 4 + y)
        lines.append(line)
        
    # Slope 2: y = 2x + c
    for c in range(4):
        line = set()
        for x in range(4):
            y = add_gf4(mul_gf4(2, x), c)
            line.add(x * 4 + y)
        lines.append(line)
        
    # Slope 3: y = 3x + c
    for c in range(4):
        line = set()
        for x in range(4):
            y = add_gf4(mul_gf4(3, x), c)
            line.add(x * 4 + y)
        lines.append(line)
        
    return lines

AG24_LINES = get_ag24_lines()

def count_4_lines(points):
    """
    Returns the number of lines containing exactly 4 points.
    Returns -1 if any line contains 5 or more points.
    """
    n = len(points)
    if n != 16:
        return -1
    
    if len(set(map(tuple, points))) != 16:
        return -1
        
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return -1

    lines = defaultdict(list)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0:
                slope = (0, 1)
            elif dy == 0:
                slope = (1, 0)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                slope = (dx_n, dy_n)
            
            A = dy
            B = -dx
            C = A*x1 + B*y1
            
            if A == 0 and B == 0:
                continue
            
            g_line = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g_line == 0:
                g_line = 1
            A_n = A // g_line
            B_n = B // g_line
            C_n = C // g_line
            
            if A_n < 0 or (A_n == 0 and B_n < 0):
                A_n = -A_n
                B_n = -B_n
                C_n = -C_n
                
            key = (A_n, B_n, C_n)
            lines[key].append(i)
            
    count_4 = 0
    for key, pts in lines.items():
        if len(pts) > 4:
            return -1
        if len(pts) == 4:
            count_4 += 1
            
    return count_4

def construct():
    # Initialize with a distorted grid
    # Try to satisfy AG24_LINES
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    # Apply a random affine transformation to break symmetries?
    # Or just use SA with a penalty for breaking AG24 lines.
    
    best_score = count_4_lines(points)
    best_points = [p[:] for p in points]
    
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    start_time = time.time()
    
    # Pre-compute indices for AG24 lines to check quickly
    # We want to maximize the number of AG24 lines that are collinear in the current config.
    # If we achieve all 20, we are likely at 20 (or more if extra collinearities appear, but hopefully not 5-collinear).
    
    def ag24_collinear_count(pts):
        count = 0
        for line in AG24_LINES:
            idxs = list(line)
            p1, p2, p3 = pts[idxs[0]], pts[idxs[1]], pts[idxs[2]]
            # Check if p4 is collinear
            p4 = pts[idxs[3]]
            # Cross product
            if (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0:
                if (p2[0]-p1[0])*(p4[1]-p1[1]) - (p2[1]-p1[1])*(p4[0]-p1[0]) == 0:
                    count += 1
        return count

    current_ag24 = ag24_collinear_count(points)
    
    while T > T_min and (time.time() - start_time) < 110:
        i = random.randint(0, 15)
        delta_x = random.randint(-5, 5)
        delta_y = random.randint(-5, 5)
        
        new_points = [p[:] for p in points]
        new_points[i][0] += delta_x
        new_points[i][1] += delta_y
        
        # Check bounds
        if abs(new_points[i][0]) > 1000 or abs(new_points[i][1]) > 1000:
            continue
            
        # Check distinctness
        if len(set(map(tuple, new_points))) != 16:
            continue
            
        new_ag24 = ag24_collinear_count(new_points)
        
        # We prefer high ag24 count.
        # If ag24 count is high, check actual score.
        delta = new_ag24 - current_ag24
        
        if delta > 0 or random.random() < math.exp(delta / T):
            points = new_points
            current_ag24 = new_ag24
            
            # If we have many AG24 lines, check the real score
            if current_ag24 > 15:
                real_score = count_4_lines(points)
                if real_score > best_score:
                    best_score = real_score
                    best_points = [p[:] for p in points]
                    
        T *= cooling
        
    # Final check on best_points
    if count_4_lines(best_points) < 0:
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
