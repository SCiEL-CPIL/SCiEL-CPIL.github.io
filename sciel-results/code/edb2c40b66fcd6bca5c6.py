import math

N = 12

def construct():
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # Ring 1
    for i in range(6):
        angle_rad = i * math.pi / 3
        x = math.sqrt(3) * math.cos(angle_rad)
        y = math.sqrt(3) * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    # Ring 2 (5 edges)
    for i in range(5):
        angle_rad = (2*i + 1) * math.pi / 6 # 30, 90, 150, 210, 270
        x = 3.0 * math.cos(angle_rad)
        y = 3.0 * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), 4.0, 30.0
