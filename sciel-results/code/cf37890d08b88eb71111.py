<<<<<<< SEARCH
def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with structured configurations
    best_points, best_score = _try_structured_configs()
    
    # Local search from best configuration
    if time.time() < deadline - 10:
        ls_points, ls_score = _local_search(best_points, deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
    
    # If still above 6, try more random restarts with local search
    if best_score > 6 and time.time() < deadline - 15:
        random.seed(42)
        for restart in range(100):
            if time.time() >= deadline - 10:
                break
            
            # Generate random configuration on 3 lines
            pts = []
            # Line 1: y = 0
            x1s = random.sample(range(-5, 8), 4)
            for x in x1s:
                pts.append([x, 0])
            
            # Line 2: y = x + c
            c2 = random.randint(-3, 3)
            x2s = random.sample(range(-5, 8), 4)
            for x in x2s:
                pt = [x, x + c2]
                if all(tuple(pt) != tuple(p) for p in pts):
                    pts.append(pt)
                else:
                    for x2 in range(-10, 10):
                        pt = [x2, x2 + c2]
                        if all(tuple(pt) != tuple(p) for p in pts):
                            pts.append(pt)
                            break
            
            # Line 3: y = mx + b
            m3 = random.choice([-2,-1,1,2,3])
            b3 = random.randint(-3, 3)
            x3s = random.sample(range(-5, 8), 4)
            for x in x3s:
                pt = [x, m3 * x + b3]
                if all(tuple(pt) != tuple(p) for p in pts):
                    pts.append(pt)
                else:
                    for x3 in range(-10, 10):
                        pt = [x3, m3 * x3 + b3]
                        if all(tuple(pt) != tuple(p) for p in pts):
                            pts.append(pt)
                            break
            
            if len(pts) != 12 or not _is_valid(pts):
                continue
            
            s = _count_ordinary_fast(pts)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
            
            # Local search
            if time.time() < deadline - 10:
                ls_points, ls_score = _local_search(pts, deadline)
                if ls_score < best_score:
                    best_score = ls_score
                    best_points = ls_points
    
    return [[int(p[0]), int(p[1])] for p in best_points]
=======
def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with structured configurations
    best_points, best_score = _try_structured_configs()
    
    # Local search from best configuration
    if time.time() < deadline - 10:
        ls_points, ls_score = _local_search(best_points, deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
    
    # If still above 6, try more random restarts with local search
    if best_score > 6 and time.time() < deadline - 15:
        random.seed(42)
        for restart in range(200):
            if time.time() >= deadline - 10:
                break
            
            # Generate random configuration on 3-4 lines
            strategy = random.choice(['3lines', '4lines', 'grid'])
            
            if strategy == '3lines':
                pts = []
                # Line 1: y = 0
                x1s = random.sample(range(-6, 9), 4)
                for x in x1s:
                    pts.append([x, 0])
                
                # Line 2: y = x + c
                c2 = random.randint(-4, 4)
                x2s = random.sample(range(-6, 9), 4)
                for x in x2s:
                    pt = [x, x + c2]
                    if all(tuple(pt) != tuple(p) for p in pts):
                        pts.append(pt)
                    else:
                        for x2 in range(-12, 12):
                            pt = [x2, x2 + c2]
                            if all(tuple(pt) != tuple(p) for p in pts):
                                pts.append(pt)
                                break
                
                # Line 3: y = mx + b
                m3 = random.choice([-3,-2,-1,1,2,3])
                b3 = random.randint(-4, 4)
                x3s = random.sample(range(-6, 9), 4)
                for x in x3s:
                    pt = [x, m3 * x + b3]
                    if all(tuple(pt) != tuple(p) for p in pts):
                        pts.append(pt)
                    else:
                        for x3 in range(-12, 12):
                            pt = [x3, m3 * x3 + b3]
                            if all(tuple(pt) != tuple(p) for p in pts):
                                pts.append(pt)
                                break
            
            elif strategy == '4lines':
                pts = []
                # 4 lines with 3 points each
                lines = []
                # Line 1: y = 0
                lines.append((0, 0))
                # Line 2: random slope/intercept
                m2 = random.choice([-2,-1,1,2,3])
                b2 = random.randint(-3, 3)
                lines.append((m2, b2))
                # Line 3: different slope
                m3 = random.choice([m for m in [-3,-2,-1,1,2,3] if m != m2])
                b3 = random.randint(-3, 3)
                lines.append((m3, b3))
                # Line 4: different slope
                m4 = random.choice([m for m in [-3,-2,-1,1,2,3] if m != m2 and m != m3])
                b4 = random.randint(-3, 3)
                lines.append((m4, b4))
                
                for m, b in lines:
                    for x in random.sample(range(-4, 7), 3):
                        pt = [x, m * x + b]
                        if all(tuple(pt) != tuple(p) for p in pts):
                            pts.append(pt)
                        else:
                            for x2 in range(-8, 8):
                                pt = [x2, m * x2 + b]
                                if all(tuple(pt) != tuple(p) for p in pts):
                                    pts.append(pt)
                                    break
            
            else:  # grid
                # Random 4x4 grid with 4 points removed
                grid = [(i, j) for i in range(4) for j in range(4)]
                removed = set(random.sample(range(16), 4))
                pts = [list(grid[i]) for i in range(16) if i not in removed]
            
            if len(pts) != 12 or not _is_valid(pts):
                continue
            
            s = _count_ordinary_fast(pts)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
            
            # Local search
            if time.time() < deadline - 10:
                ls_points, ls_score = _local_search(pts, deadline)
                if ls_score < best_score:
                    best_score = ls_score
                    best_points = ls_points
    
    return [[int(p[0]), int(p[1])] for p in best_points]
>>>>>>> REPLACE
