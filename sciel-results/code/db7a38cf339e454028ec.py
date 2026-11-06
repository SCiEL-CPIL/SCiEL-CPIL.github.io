import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 112.0

    def get_seed_32():
        """Greenwood-Gleason colouring of K32 (two copies of K16 with cross edges in colour 3)."""
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
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors

    def build_adj_color(colors, n):
        adj = [[0] * n for _ in range(4)]
        color = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                k = colors[idx]
                color[i][j] = k
                color[j][i] = k
                adj[k][i] |= (1 << j)
                adj[k][j] |= (1 << i)
                idx += 1
        return adj, color

    def count_mono_triangles(adj, n):
        total = 0
        for k in range(4):
            for i in range(n):
                m = adj[k][i]
                while m:
                    lb = m & (-m)
                    j = lb.bit_length() - 1
                    common = adj[k][i] & adj[k][j]
                    total += common.bit_count()
                    m ^= lb
        return total // 3

    def greedy_extend(adj, color, n0, n_target, rng):
        """Extend a valid n0-vertex coloring to n_target vertices."""
        adj_new = [[0] * n_target for _ in range(4)]
        color_new = [[-1] * n_target for _ in range(n_target)]

        # Copy existing
        for i in range(n0):
            for j in range(i + 1, n0):
                k = color[i][j]
                color_new[i][j] = k
                color_new[j][i] = k
                adj_new[k][i] |= (1 << j)
                adj_new[k][j] |= (1 << i)

        # Add vertices one at a time
        for nv in range(n0, n_target):
            for i in range(nv):
                best_k = -1
                best_delta = float('inf')
                for k in range(4):
                    # Count triangles (nv, i, w) where w < i and color(i,w)==k and color(nv,w)==k
                    # For each w where color(i,w)==k, if we set color(nv,i)=k, we need to check color(nv,w)
                    # But color(nv,w) is already set for w < nv... wait, nv is the new vertex, so edges (nv, w) for w < nv are being set now
                    # Actually we're setting edge (nv, i) where i < nv. Edges (nv, w) for w < i are already set.
                    # So count: for each w in [0, i) where color_new(i, w) == k, check if color_new(nv, w) == k
                    # If so, triangle (nv, i, w) is monochromatic in color k.
                    delta = 0
                    for w in range(i):
                        if color_new[i][w] == k and color_new[nv][w] == k:
                            delta += 1
                    if delta < best_delta or (delta == best_delta and rng.random() < 0.5):
                        best_delta = delta
                        best_k = k

                color_new[nv][i] = best_k
                color_new[i][nv] = best_k
                adj_new[best_k][nv] |= (1 << i)
                adj_new[best_k][i] |= (1 << nv)

        return adj_new, color_new, count_mono_triangles(adj_new, n_target)

    def run_sa(adj, color, n, time_limit, rng, T_init=2.0, T_min=0.05, cooling_rate=0.9998):
        """Simulated annealing to minimize monochromatic triangles."""
        current_score = count_mono_triangles(adj, n)
        if current_score == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(color[i][j])
            return adj, color, 0

        best_score = current_score
        best_adj = [row[:] for row in adj]
        best_color = [row[:] for row in color]

        T = T_init
        steps_since_improve = 0
        max_steps_no_improve = 20000

        while time.time() < time_limit:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            old_k = color[i][j]
            new_k = rng.randint(0, 3)
            if new_k == old_k:
                continue

            common_old = adj[old_k][i] & adj[old_k][j]
            delta_remove = common_old.bit_count()

            common_new = adj[new_k][i] & adj[new_k][j]
            delta_add = common_new.bit_count()

            delta = delta_add - delta_remove

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                adj[old_k][i] ^= (1 << j)
                adj[old_k][j] ^= (1 << i)
                adj[new_k][i] |= (1 << j)
                adj[new_k][j] |= (1 << i)
                color[i][j] = new_k
                color[j][i] = new_k
                current_score += delta

                if current_score < best_score:
                    best_score = current_score
                    best_adj = [row[:] for row in adj]
                    best_color = [row[:] for row in color]
                    steps_since_improve = 0
                    if best_score == 0:
                        colors = []
                        for i2 in range(n):
                            for j2 in range(i2 + 1, n):
                                colors.append(best_color[i2][j2])
                        return best_adj, best_color, 0
                else:
                    steps_since_improve += 1

            T *= cooling_rate
            if T < T_min:
                T = T_min

            if steps_since_improve > max_steps_no_improve:
                adj = [row[:] for row in best_adj]
                color = [row[:] for row in best_color]
                current_score = best_score
                T = T_init
                steps_since_improve = 0

        return best_adj, best_color, best_score

    def try_solve_n(n, time_limit, rng):
        """Try to find a valid coloring for K_n by extending the 32-vertex seed."""
        seed_colors = get_seed_32()
        adj0, color0 = build_adj_color(seed_colors, 32)

        # Multiple restarts with different random seeds for extension
        best_score = float('inf')
        best_adj = None
        best_color = None

        num_restarts = max(1, int(time_limit / 5.0))
        restart_time = time_limit / num_restarts

        for r in range(num_restarts):
            if time.time() > time_limit:
                break

            local_time = min(restart_time, time_limit - (time.time() - start_time) - 1)
            if local_time < 1:
                break

            # Greedy extend
            adj_ext, color_ext, score_ext = greedy_extend(adj0, color0, 32, n, rng)

            if score_ext == 0:
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        colors.append(color_ext[i][j])
                return {'n': n, 'colors': colors}

            if score_ext < best_score:
                best_score = score_ext
                best_adj = [row[:] for row in adj_ext]
                best_color = [row[:] for row in color_ext]

            # Polish with SA
            sa_time = local_time - 0.5
            if sa_time > 1:
                adj_sa, color_sa, score_sa = run_sa(adj_ext, color_ext, n, sa_time, rng)
                if score_sa < best_score:
                    best_score = score_sa
                    best_adj = [row[:] for row in adj_sa]
                    best_color = [row[:] for row in color_sa]
                    if best_score == 0:
                        colors = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                colors.append(best_color[i][j])
                        return {'n': n, 'colors': colors}

        if best_score == 0 and best_adj is not None:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(best_color[i][j])
            return {'n': n, 'colors': colors}

        return None

    rng = random.Random(42)

    # Try to find solutions for n=50 down to 33
    for n in [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]:
        if time.time() > deadline - 3:
            break

        remaining = deadline - time.time() - 1
        if remaining < 2:
            break

        # Allocate more time for larger n
        time_per_n = min(remaining * 0.85, 18.0) if n >= 45 else min(remaining * 0.85, 10.0)

        res = try_solve_n(n, time_per_n, rng)
        if res:
            return res

    # Fallback: return the 32-vertex seed
    return {'n': 32, 'colors': get_seed_32()}
