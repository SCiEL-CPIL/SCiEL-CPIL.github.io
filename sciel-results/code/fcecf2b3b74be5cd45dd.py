import time
import random
from math import gcd

def is_square(n):
    if n < 0:
        return False
    r = int(n ** 0.5)
    return r * r == n or (r + 1) * (r + 1) == n

def check_collinear(p1, p2, p3):
    return p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]) == 0

def check_concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    a1 = x1*x1 + y1*y1
    a2 = x2*x2 + y2*y2
    a3 = x3*x3 + y3*y3
    a4 = x4*x4 + y4*y4
    
    # Subtract row 4
    d11 = a1 - a4
    d12 = x1 - x4
    d13 = y1 - y4
    d21 = a2 - a4
    d22 = x2 - x4
    d23 = y2 - y4
    d31 = a3 - a4
    d32 = x3 - x4
    d33 = y3 - y4
    
    det3 = d11 * (d22 * d33 - d23 * d32) - \
           d12 * (d21 * d33 - d23 * d31) + \
           d13 * (d21 * d32 - d22 * d31)
           
    return det3 == 0

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Base set of 5 points from seed
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6th point for the base set
    # Or search for a new 6-point set using greedy growth.
    
    # Greedy Growth:
    # 1. Pick P1 = (0,0)
    # 2. Pick P2 such that d(P1,P2) is rational.
    # 3. Pick P3 such that d(P1,P3), d(P2,P3) rational, and not collinear.
    # ...
    
    # We will run multiple restarts of this greedy process.
    
    for _ in range(500): # 500 restarts
        if time.time() > deadline:
            break
            
        points = [(0, 0)]
        
        # Add P2
        # Use a Pythagorean triple (a, b, c)
        # P2 = (a, b)
        # Let's pick small ones
        triples = [(3,4,5), (5,12,13), (8,15,17), (7,24,25), (20,21,29), (12,35,37)]
        a, b, _ = random.choice(triples)
        # Randomly reflect/rotate? Just (a,b) is fine for now, or (-a,b) etc.
        P2 = (a, b)
        points.append(P2)
        
        # Add P3
        found = False
        for _ in range(1000):
            # Generate candidate
            # Distance to P1 must be rational, to P2 must be rational.
            # Let P3 = (x, y).
            # x^2 + y^2 = k1^2
            # (x-a)^2 + (y-b)^2 = k2^2
            # This is intersection of two circles.
            # Instead of solving, let's just try random integer points in a box and check.
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            P3 = (x, y)
            if P3 in points: continue
            
            d12 = x*x + y*y
            d22 = (x-a)**2 + (y-b)**2
            if is_square(d12) and is_square(d22):
                if not check_collinear(P1, P2, P3):
                    points.append(P3)
                    found = True
                    break
        if not found:
            continue
            
        # Add P4
        found = False
        for _ in range(1000):
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            P4 = (x, y)
            if P4 in points: continue
            
            ok = True
            for P in points:
                dx = x - P[0]
                dy = y - P[1]
                if not is_square(dx*dx + dy*dy):
                    ok = False
                    break
            if not ok: continue
            
            # Check collinearity with any pair
            collinear = False
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if check_collinear(points[i], points[j], P4):
                        collinear = True
                        break
                if collinear: break
            if collinear: continue
            
            # Check concyclicity with any triple
            concyclic = False
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    for k in range(j+1, len(points)):
                        if check_concyclic(points[i], points[j], points[k], P4):
                            concyclic = True
                            break
                    if concyclic: break
                if concyclic: break
            if concyclic: continue
            
            points.append(P4)
            found = True
            break
        if not found:
            continue
            
        # Add P5
        found = False
        for _ in range(1000):
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            P5 = (x, y)
            if P5 in points: continue
            
            ok = True
            for P in points:
                dx = x - P[0]
                dy = y - P[1]
                if not is_square(dx*dx + dy*dy):
                    ok = False
                    break
            if not ok: continue
            
            collinear = False
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if check_collinear(points[i], points[j], P5):
                        collinear = True
                        break
                if collinear: break
            if collinear: continue
            
            concyclic = False
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    for k in range(j+1, len(points)):
                        if check_concyclic(points[i], points[j], points[k], P5):
                            concyclic = True
                            break
                    if concyclic: break
                if concyclic: break
            if concyclic: continue
            
            points.append(P5)
            found = True
            break
        if not found:
            continue
            
        # Add P6
        found = False
        for _ in range(1000):
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            P6 = (x, y)
            if P6 in points: continue
            
            ok = True
            for P in points:
                dx = x - P[0]
                dy = y - P[1]
                if not is_square(dx*dx + dy*dy):
                    ok = False
                    break
            if not ok: continue
            
            collinear = False
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if check_collinear(points[i], points[j], P6):
                        collinear = True
                        break
                if collinear: break
            if collinear: continue
            
            concyclic = False
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    for k in range(j+1, len(points)):
                        if check_concyclic(points[i], points[j], points[k], P6):
                            concyclic = True
                            break
                    if concyclic: break
                if concyclic: break
            if concyclic: continue
            
            points.append(P6)
            found = True
            break
        if found:
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
            
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in base_points]}
