import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def precompute_c6s(n):
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                idx += 1
        total_edges = idx

        patterns = [
            [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
            [(0,1),(1,2),(2,4),(4,3),(3,5),(5,0)],
            [(0,1),(1,3),(3,2),(2,4),(4,5),(5,0)],
            [(0,1),(1,3),(3,4),(4,2),(2,5),(5,0)],
            [(0,1),(1,4),(4,2),(2,3),(3,5),(5,0)],
        ]

        c6_edge_indices = []
        for comb in combinations(range(n), 6):
            v0, v1, v2, v3, v4, v5 = comb
            for pat in patterns:
                indices = []
                for (a, b) in pat:
                    u, v = (a, b) if a < b else (b, a)
                    indices.append(edge_idx[(u, v)])
                c6_edge_indices.append(np.array(indices, dtype=np.int32))

        c6_array = np.array(c6_edge_indices, dtype=np.int32)
        num_c6s = len(c6_edge_indices)

        edge_to_c6s = [[] for _ in range(total_edges)]
        for i in range(num_c6s):
            for e in c6_array[i]:
                edge_to_c6s[e].append(i)
        edge_to_c6s_arrays = [np.array(lst, dtype=np.int32) for lst in edge_to_c6s]

        return c6_array, edge_to_c6s_arrays, total_edges

    def spectral_init(n):
        # Construct Laplacian of K_n
        # L = n*I - J
        L = n * np.eye(n) - np.ones((n, n))
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Use the first two non-trivial eigenvectors (smallest non-zero eigenvalues)
        # The smallest eigenvalue is 0 (constant vector). The next two are Fiedler vectors.
        # Sort by eigenvalue
        idx_sort = np.argsort(eigenvalues)
        # Take indices 1 and 2 (0 is the trivial one)
        # If n is small, ensure we have at least 3 eigenvalues
        vec1 = eigenvectors[:, idx_sort[1]]
        vec2 = eigenvectors[:, idx_sort[2]]
        
        # Map vertices to 2D
        points = np.column_stack((vec1, vec2))
        
        total_edges = n * (n - 1) // 2
        colors = np.zeros(total_edges, dtype=np.int32)
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Vector from i to j
                v = points[j] - points[i]
                # Angle
                angle = np.arctan2(v[1], v[0])
                # Map angle to 0..1
                norm_angle = (angle + np.pi) / (2 * np.pi)
                # Assign to 4 quadrants
                c = int(norm_angle * 4) % 4
                colors[idx] = c
                idx += 1
        return colors

    def solve(n, max_time):
        c6_array, edge_to_c6s_arrays, total_edges = precompute_c6s(n)
        num_c6s = len(c6_array)

        if num_c6s == 0:
            return np.zeros(total_edges, dtype=np.int32)

        # Precompute edge_other_edges for fast delta
        edge_other_edges = []
        for e in range(total_edges):
            c6_indices = edge_to_c6s_arrays[e]
            if len(c6_indices) == 0:
                edge_other_edges.append((c6_indices, np.empty((0, 5), dtype=np.int32)))
            else:
                all_edges = c6_array[c6_indices]
                pos = (all_edges == e).nonzero()[1]
                rows = np.arange(len(c6_indices))
                mask = np.ones((len(c6_indices), 6), dtype=bool)
                mask[rows, pos] = False
                other = all_edges[mask].reshape(len(c6_indices), 5)
                edge_other_edges.append((c6_indices, other))

        def compute_cost(col):
            c = col[c6_array]
            mono = np.all(c == c[:, 0:1], axis=1)
            return int(np.sum(mono))

        np_rng = np.random.default_rng(42)
        
        # Try multiple initializations: Spectral and Random
        best_cost = float('inf')
        best_colors = None
        
        inits = [spectral_init(n), np.random.randint(0, 4, size=total_edges, dtype=np.int32)]
        
        for init_colors in inits:
            colors = init_colors.copy()
            cost = compute_cost(colors)
            
            if cost == 0:
                return colors
            
            if cost < best_cost:
                best_cost = cost
                best_colors = colors.copy()
            
            # Simulated Annealing
            T_start = 2.0
            T_end = 0.01
            iter_count = 0
            max_iters = 500000 # Cap iterations per init to ensure we try both
            
            while time.time() < max_time and iter_count < max_iters:
                e = int(np_rng.integers(0, total_edges))
                old_color = int(colors[e])
                
                c6_indices, other_edges = edge_other_edges[e]
                k = len(c6_indices)
                
                if k == 0:
                    new_color = int(np_rng.integers(0, 4))
                    if new_color != old_color:
                        colors[e] = new_color
                    iter_count += 1
                    continue
                
                mono_old = np.all(other_edges == old_color, axis=1)
                num_mono_old = int(np.sum(mono_old))
                
                best_delta = 0
                best_new_color = -1
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    mono_new = np.all(other_edges == new_color, axis=1)
                    num_mono_new = int(np.sum(mono_new))
                    delta = num_mono_new - num_mono_old
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = new_color
                
                progress = min(1.0, iter_count / max_iters)
                T = T_start * (T_end / T_start) ** progress
                
                if best_new_color != -1:
                    if best_delta <= 0:
                        colors[e] = best_new_color
                        cost += best_delta
                        if cost < best_cost:
                            best_cost = cost
                            best_colors = colors.copy()
                            if best_cost == 0:
                                return colors
                    else:
                        if np_rng.random() < np.exp(-best_delta / T):
                            colors[e] = best_new_color
                            cost += best_delta
                            if cost < best_cost:
                                best_cost = cost
                                best_colors = colors.copy()
                                if best_cost == 0:
                                    return colors
                else:
                    # Random perturbation if no improving move
                    new_color = int(np_rng.integers(0, 4))
                    if new_color != old_color:
                        mono_new = np.all(other_edges == new_color, axis=1)
                        delta = int(np.sum(mono_new)) - num_mono_old
                        colors[e] = new_color
                        cost += delta
                        if cost < best_cost:
                            best_cost = cost
                            best_colors = colors.copy()
                            if best_cost == 0:
                                return colors
                            
                iter_count += 1
                
                # Restart logic
                if iter_count % 50000 == 0 and best_cost > 0:
                    if cost > best_cost * 1.5 + 10:
                        colors = best_colors.copy()
                        cost = best_cost
                        num_flips = max(1, total_edges // 10)
                        flip_indices = np_rng.choice(total_edges, size=num_flips, replace=False)
                        for fi in flip_indices:
                            colors[fi] = int(np_rng.integers(0, 4))
                        cost = compute_cost(colors)
                        if cost < best_cost:
                            best_cost = cost
                            best_colors = colors.copy()
                            if best_cost == 0:
                                return colors

        if best_cost == 0:
            return best_colors
        return None

    # Try n=17
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol = solve(17, remaining - 2.0)
        if sol is not None:
            return {'n': 17, 'colors': sol.tolist()}

    # Try n=16
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol = solve(16, remaining - 2.0)
        if sol is not None:
            return {'n': 16, 'colors': sol.tolist()}

    # Try n=15
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol = solve(15, remaining - 2.0)
        if sol is not None:
            return {'n': 15, 'colors': sol.tolist()}

    # Fallback: n=1
    return {'n': 1, 'colors': []}
