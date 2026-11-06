import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Try n from 26 down to 11
    for n in range(26, 10, -1):
        if time.time() > deadline - 30:
            break

        # Random restarts
        # More restarts for smaller n, fewer for larger n
        max_restarts = 100 if n <= 20 else 20
        found = False
        for r in range(max_restarts):
            if time.time() > deadline - 10:
                break

            # Initialize
            m = n * (n - 1) // 2
            colors = np.random.randint(0, 5, size=m)

            # Build neighbor lists
            adj = [ [ [] for _ in range(n) ] for _ in range(5) ]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i].append(j)
                    adj[c][j].append(i)
                    idx += 1

            # common_counts[c][u][v]
            common_counts = [ np.zeros((n, n), dtype=np.int8) for _ in range(5) ]

            # Calculate initial common counts
            for c in range(5):
                for u in range(n):
                    # Use set intersection for speed
                    set_u = set(adj[c][u])
                    for v in range(n):
                        if u == v: continue
                        set_v = set(adj[c][v])
                        common_counts[c][u][v] = len(set_u & set_v)

            # Calculate total violations
            violations = 0
            for c in range(5):
                # Sum max(0, count - 1)
                # We can use numpy
                violations += np.sum(np.maximum(0, common_counts[c] - 1))

            # Local search
            col_2d = np.zeros((n, n), dtype=np.int8)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    col_2d[i][j] = colors[idx]
                    col_2d[j][i] = colors[idx]
                    idx += 1

            no_improve_count = 0
            max_no_improve = 500
            while violations > 0 and no_improve_count < max_no_improve:
                if time.time() > deadline - 5:
                    break

                # Pick a random edge
                u = np.random.randint(n)
                v = np.random.randint(n)
                if u == v: continue
                if u > v: u, v = v, u

                c_old = col_2d[u][v]

                best_delta = 0
                best_c = c_old

                for c_new in range(5):
                    if c_new == c_old: continue

                    delta = 0

                    # c_old updates
                    # For w in adj[c_old][v]:
                    #   common_counts[c_old][u][w] decreases by 1
                    #   If it was >= 2, violation decreases by 1
                    for w in adj[c_old][v]:
                        if common_counts[c_old][u][w] >= 2:
                            delta -= 1
                    for w in adj[c_old][u]:
                        if common_counts[c_old][v][w] >= 2:
                            delta -= 1

                    # c_new updates
                    # For w in adj[c_new][v]:
                    #   common_counts[c_new][u][w] increases by 1
                    #   If it was 1, violation increases by 1
                    for w in adj[c_new][v]:
                        if common_counts[c_new][u][w] == 1:
                            delta += 1
                    for w in adj[c_new][u]:
                        if common_counts[c_new][v][w] == 1:
                            delta += 1

                    if delta < best_delta:
                        best_delta = delta
                        best_c = c_new

                if best_c != c_old:
                    # Apply move
                    adj[c_old][u].remove(v)
                    adj[c_old][v].remove(u)
                    adj[best_c][u].append(v)
                    adj[best_c][v].append(u)

                    for w in adj[c_old][v]:
                        common_counts[c_old][u][w] -= 1
                        common_counts[c_old][w][u] -= 1
                    for w in adj[c_old][u]:
                        common_counts[c_old][v][w] -= 1
                        common_counts[c_old][w][v] -= 1

                    for w in adj[best_c][v]:
                        common_counts[best_c][u][w] += 1
                        common_counts[best_c][w][u] += 1
                    for w in adj[best_c][u]:
                        common_counts[best_c][v][w] += 1
                        common_counts[best_c][w][v] += 1

                    violations += best_delta
                    col_2d[u][v] = best_c
                    col_2d[v][u] = best_c
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            if violations == 0:
                flat_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        flat_colors.append(int(col_2d[i][j]))
                return {'n': n, 'colors': flat_colors}

    # Fallback to n=11
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
