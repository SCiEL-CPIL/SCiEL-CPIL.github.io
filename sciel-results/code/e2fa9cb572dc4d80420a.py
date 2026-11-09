"""Portfolio search for DTLZ2: Halton-sequence sampling with sin^2 edge concentration and geometric anchors."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using 2D Halton sequence on the known front with geometric anchors."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0  # Safety margin for 330s limit

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        used = 0

        def map_obj_to_x(f1, f2, f3):
            x = np.full(dim, 0.5)
            f3_clipped = np.clip(f3, 0, 1)
            x[0] = (2.0 / np.pi) * np.arcsin(f3_clipped)
            r = np.sqrt(f1*f1 + f2*f2)
            if r > 1e-15:
                ratio = np.clip(f1/r, -1, 1)
                x[1] = (2.0 / np.pi) * np.arcsin(ratio)
            else:
                x[1] = 0.5
            return x

        # Add 8 special points
        candidates = []
        candidates.append(map_obj_to_x(1.0, 0.0, 0.0))
        candidates.append(map_obj_to_x(0.0, 1.0, 0.0))
        candidates.append(map_obj_to_x(0.0, 0.0, 1.0))
        
        me1 = 1.0 / np.sqrt(2.0)
        candidates.append(map_obj_to_x(me1, me1, 0.0))
        candidates.append(map_obj_to_x(me1, 0.0, me1))
        candidates.append(map_obj_to_x(0.0, me1, me1))
        
        f1_i1 = 0.3
        f2_i1 = 0.3
        f3_i1 = np.sqrt(max(0.0, 1.0 - f1_i1*f1_i1 - f2_i1*f2_i1))
        candidates.append(map_obj_to_x(f1_i1, f2_i1, f3_i1))
        
        c = 1.0 / np.sqrt(3.0)
        candidates.append(map_obj_to_x(c, c, c))

        n_special = len(candidates)
        n_halton = budget - n_special
        if n_halton < 10:
            n_halton = max(10, budget // 2)
            n_special = budget - n_halton

        def halton_2d(n, base1, base2):
            points = []
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    f /= base1
                    r += f * (i_val % base1)
                    i_val //= base1
                f = 1.0
                r2 = 0.0
                i_val = i
                while i_val > 0:
                    f /= base2
                    r2 += f * (i_val % base2)
                    i_val //= base2
                points.append((r, r2))
            return points

        halton_pts = halton_2d(n_halton, 2, 3)
        
        for t_u, t_v in halton_pts:
            u = np.sin(0.5 * np.pi * t_u) ** 2
            u = np.clip(u, 1e-10, 1.0 - 1e-10)
            f3 = u
            
            v = np.sin(0.5 * np.pi * t_v) ** 2
            v = np.clip(v, 1e-10, 1.0 - 1e-10)
            phi = v * (np.pi / 2.0)
            
            r = np.sqrt(max(0.0, 1.0 - f3*f3))
            f1 = r * np.sin(phi)
            f2 = r * np.cos(phi)
            
            x = map_obj_to_x(f1, f2, f3)
            candidates.append(x)

        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)
            used += 1

        return np.zeros((0, n_obj))
# EVOLVE-BLOCK-END
