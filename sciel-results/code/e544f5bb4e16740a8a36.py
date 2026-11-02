import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110.0
    
    # 1. Try to construct the (3,16)-cage as a lift of the Foster graph (3,12)-cage.
    # The Foster graph has 156 vertices. A lift of size 6 has 936 vertices.
    # We search for a voltage assignment over Z6 that yields girth >= 16.
    
    # Foster Graph Edges (156 vertices, 234 edges)
    # This is the (3,12)-cage.
    # I will generate the edges using a known representation or hardcode them.
    # To save space, I'll use a compact representation if possible, or just hardcode.
    # Hardcoding is safer for correctness.
    
    # Note: I don't have the exact edge list memorized. I will generate a random 3-regular graph 
    # and check if it's isomorphic to Foster? No, that's hard.
    # Alternative: Use the LUW construction but with a twist?
    # Or, use a different known graph.
    
    # Actually, if I don't have the Foster edges, I can't do this easily.
    # Let's check if I can generate the Foster graph.
    # The Foster graph is the unique (3,12)-cage.
    # It can be constructed as a voltage graph over a smaller base?
    
    # If I cannot hardcode the edges, I should fall back to LUW.
    # But wait, I can try to find the edges online in my training data?
    # I recall the Foster graph is related to the (3,12) cage.
    
    # Let's try a different approach: 
    # Is there a smaller explicit graph?
    # The (3,16) cage is 936.
    # LUW is ~1M.
    # If I can't get 936, LUW is the best I have.
    
    # However, I can optimize the LUW construction to be faster, 
    # but the score is n.
    
    # Let's check if there is a bug in the LUW size calculation.
    # n = 2 * q^(d+1).
    # For d=11, q=3, n = 2 * 3^12 = 1062882.
    
    # Is there a smaller d?
    # Girth >= d+5.
    # If d=9, girth >= 14.
    # If d=10, girth >= 15? (d must be odd for the standard LUW formula? 
    # The formula is for odd d. For even d, it might be different.)
    # If d=10, girth >= 15. Not enough.
    
    # So d=11 is the minimum for LUW.
    
    # Conclusion: I will stick to LUW but ensure the code is robust.
    # I will also add a quick check for small g to return known small cages if g < 16?
    # The task says "evaluator passes this task's degree k and girth g".
    # It implies k=3, g=16.
    
    # I will just return the LUW graph.
    
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    rule_list = [None] * (d + 1)
    for c in range(2, d + 1):
        rule_list[c] = rule[c]
    
    start = (0, tuple([None] + [0] * d))
    
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        side, x = vertex
        for first in range(q):
            y_coords = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule_list[c]
                if side == 0:
                    y_coords[c] = add(x[c], mul(y_coords[a], x[b]))
                else:
                    y_coords[c] = sub(x[c], mul(x[a], y_coords[b]))
            other = (1 - side, tuple(y_coords))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
