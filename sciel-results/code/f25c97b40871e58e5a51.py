import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s kill

    # 1. Construct base K32 coloring (Greenwood-Gleason / Clebsch doubling)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            base_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)

    current_n = 32
    current_colors = base_colors

    # Helper to get color of edge (i, j) from flat list
    def get_offsets(n):
        offsets = [0] * n
        acc = 0
        for i in range(n):
            offsets[i] = acc
            acc += n - i - 1
        return offsets

    def try_extend(colors, n, seed=None):
        """
        Try to extend K_n to K_{n+1} using backtracking CSP.
        Returns new colors list or None.
        """
        if seed is not None:
            random.seed(seed)

        # Build adjacency lists for each color k
        adj = [[[] for _ in range(n)] for _ in range(4)]
        offsets = get_offsets(n)

        for i in range(n):
            for j in range(i + 1, n):
                c = colors[offsets[i] + j - i - 1]
                adj[c][i].append(j)
                adj[c][j].append(i)

        c = [-1] * n
        forbidden = [0] * n

        # Precompute variable order (MRV heuristic can be applied dynamically)
        # For speed, we use a static order but with forward checking

        def backtrack(idx, depth=0):
            if idx == n:
                return True

            # Find unassigned variable with fewest available colors (MRV)
            min_var = -1
            min_count = 5
            for v in range(n):
                if c[v] != -1:
                    continue
                avail = 15 & ~forbidden[v]
                count = bin(avail).count('1')
                if count < min_count:
                    min_count = count
                    min_var = v
                    if count == 0:
                        return False
            if min_var == -1:
                return True

            v = min_var
            avail = 15 & ~forbidden[v]

            # Value ordering: try colors that minimize constraints
            # For simplicity, try 0-3, but shuffle if needed
            color_order = list(range(4))
            if seed is not None:
                random.shuffle(color_order)

            for k in color_order:
                if not (avail & (1 << k)):
                    continue

                c[v] = k
                # Propagate: for all j in adj[k][v], forbid color k for c_j
                new_forbidden = []
                valid = True
                for j in adj[k][v]:
                    if c[j] == k:
                        valid = False
                        break
                    if c[j] == -1:
                        if not (forbidden[j] & (1 << k)):
                            new_forbidden.append((j, forbidden[j] | (1 << k)))

                if not valid:
                    c[v] = -1
                    continue

                # Apply changes
                old_forbidden = []
                for j, f in new_forbidden:
                    old_forbidden.append((j, forbidden[j]))
                    forbidden[j] = f
                    if forbidden[j] == 15:
                        valid = False
                        break

                if valid:
                    if backtrack(idx + 1, depth + 1):
                        return True

                # Undo
                for j, old_f in old_forbidden:
                    forbidden[j] = old_f
                c[v] = -1

            return False

        if backtrack(0):
            new_colors = colors[:]
            for i in range(n):
                new_colors.append(c[i])
            return new_colors
        return None

    # Iterative extension with multiple attempts and random restarts
    while current_n < 51 and time.time() < deadline:
        success = False
        attempts = 0
        max_attempts = 20

        while not success and attempts < max_attempts and time.time() < deadline:
            seed = None if attempts == 0 else attempts * 12345 + 67890
            new_colors = try_extend(current_colors, current_n, seed)
            if new_colors:
                current_colors = new_colors
                current_n += 1
                success = True
            else:
                attempts += 1

        if not success:
            # If we can't extend further, break
            break

    return {'n': current_n, 'colors': current_colors}
