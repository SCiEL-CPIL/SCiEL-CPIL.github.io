import random
import time

def construct():
    random.seed(42)
    
    LB, UB = -100, 100
    N = 20
    
    def is_collinear(p1, p2, p3):
        return p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]) == 0

    def is_on_line(p, p1, p2):
        # Check if p is on line defined by p1, p2
        # Assumes p1 != p2
        return (p1[0]-p[0])*(p2[1]-p1[1]) - (p2[0]-p1[0])*(p1[1]-p[1]) == 0

    def check_validity_no4(points):
        # Full check for initial generation
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        # Check if 4th point exists
                        # Line eq
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        A = y1 - y2
                        B = x2 - x1
                        C = x1*y2 - x2*y1
                        count = 0
                        for m in range(n):
                            if A*points[m][0] + B*points[m][1] + C == 0:
                                count += 1
                        if count >= 4:
                            return False
        return True

    def count_triples_involving(p, others):
        # Count triples (p, o1, o2) that are collinear
        # others is a list of points
        count = 0
        n = len(others)
        for i in range(n):
            for j in range(i+1, n):
                if is_collinear(p, others[i], others[j]):
                    count += 1
        return count

    def generate_initial():
        while True:
            pts = [[random.randint(LB, UB), random.randint(LB, UB)] for _ in range(N)]
            if len(set(map(tuple, pts))) < N:
                continue
            if check_validity_no4(pts):
                return pts

    start_time = time.time()
    current_pts = generate_initial()
    
    # Calculate initial total triples
    total_triples = 0
    for i in range(N):
        others = current_pts[:i] + current_pts[i+1:]
        total_triples += count_triples_involving(current_pts[i], others)
    # Each triple is counted 3 times (once for each vertex)
    total_triples //= 3
    
    best_pts = current_pts[:]
    best_triples = total_triples
    
    steps = 100000
    for step in range(steps):
        if time.time() - start_time > 100:
            break
            
        i = random.randint(0, N-1)
        old_p = current_pts[i]
        others = current_pts[:i] + current_pts[i+1:]
        
        new_p = [random.randint(LB, UB), random.randint(LB, UB)]
        if new_p == old_p or new_p in others:
            continue
            
        # Check validity: 
        # 1. Does new_p form a 4-collinear with any 3 points in others?
        valid = True
        # Iterate triples in others
        # If any triple is collinear, check if new_p is on that line
        n_others = len(others)
        for a in range(n_others):
            for b in range(a+1, n_others):
                for c in range(b+1, n_others):
                    if is_collinear(others[a], others[b], others[c]):
                        # Check if new_p is on this line
                        if is_on_line(new_p, others[a], others[b]):
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Calculate change in triples
        # Old triples involving old_p
        old_contrib = count_triples_involving(old_p, others)
        # New triples involving new_p
        new_contrib = count_triples_involving(new_p, others)
        
        delta = new_contrib - old_contrib
        
        if delta > 0:
            current_pts[i] = new_p
            total_triples += delta
            if total_triples > best_triples:
                best_triples = total_triples
                best_pts = current_pts[:]
                
    return best_pts
