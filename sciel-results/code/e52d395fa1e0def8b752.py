import time

def construct():
    # 8 disjoint copies of K4: n=32, no 5-cycle, independence number = 8 < 9
    edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    return {'n': 32, 'edges': edges}
