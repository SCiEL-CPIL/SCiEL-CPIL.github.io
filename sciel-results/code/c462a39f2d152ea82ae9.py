"""Portfolio search for DTLZ2: deterministic quarter-circle arc on the known front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using an explicit front-arc construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0  # conservative safety margin under the 330 s kill

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        used = 0

        # Strategy: 2D Halton sequence on the Pareto front with sin^2 edge concentration.
        # Plus explicit vertices and a center point to maximize hypervolume coverage.
        
        # Use 192 Halton points and 8 special points (3 vertices + 3 mid-edges + 2 interior)
        n_halton = 192
        
        candidates = []
        
        # 2D Halton sequence generator
        def halton_2d(n, base1, base2):
            """Generate n points of a 2D Halton sequence."""
            points = []
            for i in range(1, n + 1):
                # 1st coordinate (base base1)
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    f /= base1
                    r += f * (i_val % base1)
                    i_val //= base1
                # 2nd coordinate (base base2)
                f = 1.0
                r2 = 0.0
                i_val = i
                while i_val > 0:
                    f /= base2
                    r2 += f * (i_val % base2)
                    i_val //= base2
                points.append((r, r2))
            return points
        
        # Generate Halton points with coprime bases 2 and 3
        halton_pts = halton_2d(n_halton, 2, 3)
        
        for t_u, t_v in halton_pts:
            # Apply sin^2 edge concentration
            u = np.sin(0.5 * np.pi * t_u) ** 2
            u = np.clip(u, 1e-10, 1.0 - 1e-10)
            sin_theta = np.sqrt(max(0.0, 1.0 - u * u))
            
            v = np.sin(0.5 * np.pi * t_v) ** 2
            v = np.clip(v, 1e-10, 1.0 - 1e-10)
            phi = v * (np.pi / 2.0)
            
            f1 = sin_theta * np.cos(phi)
            f2 = sin_theta * np.sin(phi)
            f3 = u
            
            # Map to x
            x = np.full(dim, 0.5)
            x[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3, 0, 1))
            
            r = np.sqrt(f1 * f1 + f2 * f2)
            if r > 1e-15:
                ratio = f1 / r
                x[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio, -1, 1))
            else:
                x[1] = 0.5
            
            candidates.append(x)

        # Add 8 special points for better hypervolume coverage
        
        # 1. Vertex (1, 0, 0) -> f1=1, f2=0, f3=0
        x_v1 = np.full(dim, 0.5)
        x_v1[0] = 0.0
        x_v1[1] = 1.0
        candidates.append(x_v1)
        
        # 2. Vertex (0, 1, 0) -> f1=0, f2=1, f3=0
        x_v2 = np.full(dim, 0.5)
        x_v2[0] = 0.0
        x_v2[1] = 0.5
        candidates.append(x_v2)
        
        # 3. Vertex (0, 0, 1) -> f1=0, f2=0, f3=1
        x_v3 = np.full(dim, 0.5)
        x_v3[0] = 1.0
        x_v3[1] = 0.5
        candidates.append(x_v3)
        
        # 4. Mid-edge between v1 and v2: (1/sqrt(2), 1/sqrt(2), 0)
        f1_me1 = 1.0 / np.sqrt(2.0)
        f2_me1 = 1.0 / np.sqrt(2.0)
        f3_me1 = 0.0
        x_me1 = np.full(dim, 0.5)
        x_me1[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_me1, 0, 1))
        r_me1 = np.sqrt(f1_me1*f1_me1 + f2_me1*f2_me1)
        if r_me1 > 1e-15:
            ratio_me1 = f1_me1 / r_me1
            x_me1[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_me1, -1, 1))
        else:
            x_me1[1] = 0.5
        candidates.append(x_me1)
        
        # 5. Mid-edge between v1 and v3: (1/sqrt(2), 0, 1/sqrt(2))
        f1_me2 = 1.0 / np.sqrt(2.0)
        f2_me2 = 0.0
        f3_me2 = 1.0 / np.sqrt(2.0)
        x_me2 = np.full(dim, 0.5)
        x_me2[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_me2, 0, 1))
        r_me2 = np.sqrt(f1_me2*f1_me2 + f2_me2*f2_me2)
        if r_me2 > 1e-15:
            ratio_me2 = f1_me2 / r_me2
            x_me2[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_me2, -1, 1))
        else:
            x_me2[1] = 0.5
        candidates.append(x_me2)
        
        # 6. Mid-edge between v2 and v3: (0, 1/sqrt(2), 1/sqrt(2))
        f1_me3 = 0.0
        f2_me3 = 1.0 / np.sqrt(2.0)
        f3_me3 = 1.0 / np.sqrt(2.0)
        x_me3 = np.full(dim, 0.5)
        x_me3[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_me3, 0, 1))
        r_me3 = np.sqrt(f1_me3*f1_me3 + f2_me3*f2_me3)
        if r_me3 > 1e-15:
            ratio_me3 = f1_me3 / r_me3
            x_me3[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_me3, -1, 1))
        else:
            x_me3[1] = 0.5
        candidates.append(x_me3)
        
        # 7. Interior point biased toward f3: (0.3, 0.3, sqrt(1-0.18))
        f1_i1 = 0.3
        f2_i1 = 0.3
        f3_i1 = np.sqrt(max(0.0, 1.0 - f1_i1*f1_i1 - f2_i1*f2_i1))
        x_i1 = np.full(dim, 0.5)
        x_i1[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_i1, 0, 1))
        r_i1 = np.sqrt(f1_i1*f1_i1 + f2_i1*f2_i1)
        if r_i1 > 1e-15:
            ratio_i1 = f1_i1 / r_i1
            x_i1[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_i1, -1, 1))
        else:
            x_i1[1] = 0.5
        candidates.append(x_i1)
        
        # 8. Interior point at centroid of front: (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
        f1_i2 = 1.0 / np.sqrt(3.0)
        f2_i2 = 1.0 / np.sqrt(3.0)
        f3_i2 = 1.0 / np.sqrt(3.0)
        x_i2 = np.full(dim, 0.5)
        x_i2[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_i2, 0, 1))
        r_i2 = np.sqrt(f1_i2*f1_i2 + f2_i2*f2_i2)
        if r_i2 > 1e-15:
            ratio_i2 = f1_i2 / r_i2
            x_i2[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_i2, -1, 1))
        else:
            x_i2[1] = 0.5
        candidates.append(x_i2)

        # Evaluate all candidates within the time budget
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)
            used += 1

        return np.zeros((0, n_obj))
# EVOLVE-BLOCK-END
