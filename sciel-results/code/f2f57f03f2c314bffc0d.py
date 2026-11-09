import random
import time

def construct():
    n = 128
    start_time = time.time()
    
    # Precompute all 4-term APs
    aps = []
    for a in range(1, n+1):
        for d in range(1, (n - a) // 3 + 1):
            aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
    
    for k in [3, 4, 5, 6]:
        for attempt in range(300):
            if time.time() - start_time > 50:
                break
            
            colors = [random.randint(0, k-1) for _ in range(n)]
            
            for it in range(3000):
                if time.time() - start_time > 55:
                    break
                
                # Find violations
                violations = []
                for idx, (p0, p1, p2, p3) in enumerate(aps):
                    c = {colors[p0], colors[p1], colors[p2], colors[p3]}
                    if len(c) < 3:
                        violations.append((p0, p1, p2, p3))
                
                if not violations:
                    return colors
                
                # Fix a random violation
                p0, p1, p2, p3 = random.choice(violations)
                idx = random.choice([p0, p1, p2, p3])
                colors[idx] = random.randint(0, k-1)
    
    # Fallback: greedy
    colors = []
    for x in range(1, n+1):
        for color in range(128):
            valid = True
            for d in range(1, (x-1)//3 + 1):
                if x - 3*d < 1:
                    break
                prev_colors = {colors[x-1-3*d], colors[x-1-2*d], colors[x-1-d]}
                if len(prev_colors | {color}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
    
    return colors
