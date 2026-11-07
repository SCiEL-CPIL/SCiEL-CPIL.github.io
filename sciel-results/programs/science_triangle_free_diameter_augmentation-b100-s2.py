import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Hub 11
    # Vertices 9, 10, 11, 12, 13 are within distance 2 of 11 via path edges.
    # Remaining vertices: 0-8 (P9) and 14-23 (P10).
    # We need an Independent Dominating Set for P9 and P10.
    
    # For P9 (0-8), IDS is {1, 4, 7}
    # 1 covers 0,1,2
    # 4 covers 3,4,5
    # 7 covers 6,7,8
    # Independent: 1,4,7 are not adjacent.
    
    # For P10 (14-23), mapped to 0-9.
    # IDS is {1, 4, 7, 9} in local coords -> {15, 18, 21, 23} in global.
    # 15 covers 14,15,16
    # 18 covers 17,18,19
    # 21 covers 20,21,22
    # 23 covers 22,23
    # Independent: 15,18,21,23 are not adjacent.
    # 15-18 (dist 3), 18-21 (dist 3), 21-23 (dist 2, not adjacent).
    
    hub = 11
    targets = [1, 4, 7, 15, 18, 21, 23]
    
    added_edges = []
    for t in targets:
        added_edges.append([hub, t])
        
    return added_edges
