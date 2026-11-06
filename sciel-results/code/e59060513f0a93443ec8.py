import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def _is_valid(coloring, n):
        # Slow check for monochromatic C6
        adj = [[[] for _ in range(n)] for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = coloring[idx]
                adj[c][i].append(j)
                adj[c][j].append(i)
                idx += 1
        
        for c in range(4):
            for start in range(n):
                # Check for a cycle of length 6 starting at start
                # Use DFS with backtracking
                stack = [(start, [start], 0)]
                while stack:
                    u, path, d = stack.pop()
                    if d == 5:
                        if start in adj[c][u]:
                            return False
                        continue
                    if d < 5:
                        for v in adj[c][u]:
                            if v not in path:
                                stack.append((v, path + [v], d + 1))
        return True

    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
        for _ in range(1000):
            if time.time() > deadline:
                break
            coloring = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            if _is_valid(coloring, n):
                return {'n': n, 'colors': coloring}
    
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
