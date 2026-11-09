"""Construct edge coloring of K16 with no rainbow 4-cycle, maximizing distinct colors."""
import time
import itertools

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Partition vertices into 4 parts of size 4
    # Parts: {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15}
    parts = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    # Build edge index mapping: (i,j) with i<j -> index in lexicographic order
    # Lexicographic order of pairs (i,j), 0<=i<j<16
    edge_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_to_idx[(i,j)] = idx
            idx += 1
    
    colors = [0] * (N*(N-1)//2)
    
    # Assign colors:
    # - Within each part: 6 distinct colors (K4 has no 4-cycles)
    # - Between parts: one color per pair of parts (6 pairs)
    
    color_counter = 0
    
    # Internal edges of each part
    for part in parts:
        for a, b in itertools.combinations(part, 2):
            i, j = min(a,b), max(a,b)
            colors[edge_to_idx[(i,j)]] = color_counter
            color_counter += 1
    
    # Between-part edges: one color per pair of parts
    for p1, p2 in itertools.combinations(range(4), 2):
        for u in parts[p1]:
            for v in parts[p2]:
                i, j = min(u,v), max(u,v)
                colors[edge_to_idx[(i,j)]] = color_counter
        color_counter += 1
    
    # Verify no rainbow 4-cycle (quick check)
    # A 4-cycle on vertices a,b,c,d has edges ab,bc,cd,da (or ab,bd,dc,ca etc.)
    # Actually, a 4-cycle in K4 on vertices {a,b,c,d} can be:
    # (a,b,c,d): edges ab,bc,cd,da
    # (a,b,d,c): edges ab,bd,dc,ca
    # (a,c,b,d): edges ac,cb,bd,da
    # We need to check all three 4-cycles on every 4-vertex subset
    
    def get_color(u, v):
        i, j = min(u,v), max(u,v)
        return colors[edge_to_idx[(i,j)]]
    
    valid = True
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on {a,b,c,d}:
        # Cycle 1: a-b-c-d-a
        # Cycle 2: a-b-d-c-a
        # Cycle 3: a-c-b-d-a
        cycles = [
            [(a,b), (b,c), (c,d), (d,a)],
            [(a,b), (b,d), (d,c), (c,a)],
            [(a,c), (c,b), (b,d), (d,a)]
        ]
        for cycle in cycles:
            cyc_colors = set()
            for (u,v) in cycle:
                cyc_colors.add(get_color(u,v))
            if len(cyc_colors) == 4:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: all zeros
        return [0] * (N*(N-1)//2)
    
    return colors

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
