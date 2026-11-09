"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using HV-guided search with adaptive mutation and exact 3D HV."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _hv_3d(self, F, ref):
        """Compute exact hypervolume for 3D points strictly below ref."""
        valid = np.all(F < ref, axis=1)
        Fv = F[valid]
        n = len(Fv)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(np.maximum(0, ref - Fv[0])))
        
        # Sort by first objective ascending
        order = np.argsort(Fv[:, 0])
        Fv = Fv[order]
        
        # Remove dominated points (keep Pareto front)
        keep = np.ones(n, dtype=bool)
        for i in range(n - 1):
            if Fv[i, 1] <= Fv[i+1, 1] and Fv[i, 2] <= Fv[i+1, 2]:
                keep[i+1] = False
        Fv = Fv[keep]
        n = len(Fv)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(np.maximum(0, ref - Fv[0])))
        
        # Standard 3D HV algorithm:
        # For each point i in sorted order, compute the volume of the box
        # that it uniquely contributes. The box is defined by:
        # [Fv[i,0], Fv[i+1,0]) x [min(Fv[i,1], Fv[i+1,1]), ref[1]) x [min(Fv[i,2], Fv[i+1,2]), ref[2])
        # Wait, that's not quite right. Let's use the standard incremental approach.
        
        # Actually, the standard efficient algorithm for 3D is:
        # Sort by f1. For each point i, the contribution is:
        # (f1[i+1] - f1[i]) * (ref[1] - min(f2[i], f2[i+1])) * (ref[2] - min(f2[i], f2[i+1]))? No.
        
        # Let's use the recursive or sweep-line approach.
        # A simpler correct method for small n (n <= 200):
        # For each point i, the volume it contributes is the volume of the box
        # [Fv[i,0], ref[0]) x [Fv[i,1], ref[1]) x [Fv[i,2], ref[2])
        # minus the volume of boxes from points that dominate it in the relevant sub-space.
        
        # Since we have the Pareto front sorted by f1, for point i,
        # the points that can overlap with its box are i+1, i+2, ...
        # But only those with f2 < Fv[i,1] or f3 < Fv[i,2] can overlap.
        
        # Actually, let's use the standard algorithm:
        # HV = sum over i of (f1[i+1] - f1[i]) * HV_2D(points i+1..n, ref[1:])
        # where HV_2D is the 2D hypervolume of the points (f2, f3) for points with f1 < f1[i+1]
        
        # This is O(n^2) for 3D, which is fine for n=200.
        
        hv = 0.0
        for i in range(n - 1):
            # Points from i+1 to n-1 have f1 > Fv[i,0]
            # The width in f1 is Fv[i+1,0] - Fv[i,0]
            # The cross-section is the 2D HV of points i+1..n-1 in (f2, f3) space
            # But we only count the part that is "new" relative to point i
            # Actually, the standard formula is:
            # HV = sum_{i=0}^{n-2} (f1[i+1] - f1[i]) * HV_2D({(f2[j], f3[j]) for j=i+1..n-1}, ref[1:])
            #     + (ref[0] - f1[n-1]) * HV_2D({(f2[n-1], f3[n-1])}, ref[1:])
            
            # Let's implement HV_2D
            def hv_2d(points, ref2d):
                # points: list of (x, y)
                # ref2d: (ref_x, ref_y)
                if len(points) == 0:
                    return 0.0
                if len(points) == 1:
                    return max(0, ref2d[0] - points[0][0]) * max(0, ref2d[1] - points[0][1])
                
                # Sort by x ascending
                pts = sorted(points, key=lambda p: p[0])
                # Remove dominated points
                keep = [pts[0]]
                for i in range(1, len(pts)):
                    if pts[i][1] < keep[-1][1]:
                        keep.append(pts[i])
                if len(keep) == 0:
                    return 0.0
                if len(keep) == 1:
                    return max(0, ref2d[0] - keep[0][0]) * max(0, ref2d[1] - keep[0][1])
                
                hv2 = 0.0
                for i in range(len(keep) - 1):
                    width = keep[i+1][0] - keep[i][0]
                    height = ref2d[1] - keep[i][1]
                    hv2 += width * height
                # Last point
                width = ref2d[0] - keep[-1][0]
                height = ref2d[1] - keep[-1][1]
                hv2 += width * height
                return hv2
            
            # Get points from i+1 to n-1
            sub_points = [(Fv[j, 1], Fv[j, 2]) for j in range(i + 1, n)]
            hv2 = hv_2d(sub_points, ref[1:])
            width = Fv[i+1, 0] - Fv[i, 0]
            hv += width * hv2
        
        # Last point
        hv2 = hv_2d([(Fv[n-1, 1], Fv[n-1, 2])], ref[1:])
        width = ref[0] - Fv[n-1, 0]
        hv += width * hv2
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref = np.array([1705.0, 12.0, 0.3])

        # Initial random sampling
        n_init = min(40, self.budget // 5)
        X_list = []
        F_list = []

        for i in range(n_init):
            x = lo + span * rng.random(self.dim)
            f = np.array(problem(x))
            X_list.append(x)
            F_list.append(f)

        used = n_init
        step_size = 0.25

        while used < self.budget:
            if time.time() > deadline:
                break

            F_arr = np.array(F_list)
            n = len(F_arr)

            # Compute exact HV contribution for each point
            valid = np.all(F_arr < ref, axis=1)
            if np.sum(valid) == 0:
                parent_idx = rng.integers(n)
            else:
                # Compute total HV
                total_hv = self._hv_3d(F_arr, ref)
                
                # Compute contribution of each point by removing it
                contrib = np.zeros(n)
                valid_indices = np.where(valid)[0]
                
                for idx in valid_indices:
                    # Remove point idx and compute HV
                    mask = np.ones(n, dtype=bool)
                    mask[idx] = False
                    F_reduced = F_arr[mask]
                    hv_reduced = self._hv_3d(F_reduced, ref)
                    contrib[idx] = total_hv - hv_reduced
                
                if np.max(contrib) > 0:
                    top_k = min(5, int(np.sum(contrib > 0)))
                    if top_k > 0:
                        valid_idx = np.where(contrib > 0)[0]
                        top_idx = valid_idx[np.argsort(contrib[valid_idx])][-top_k:]
                        w = contrib[top_idx]
                        w = w / w.sum()
                        parent_idx = rng.choice(top_idx, p=w)
                    else:
                        parent_idx = rng.integers(n)
                else:
                    parent_idx = rng.integers(n)

            parent = X_list[parent_idx]
            child = parent + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = np.array(problem(child))
            X_list.append(child)
            F_list.append(fc)
            used += 1

            # Adapt step size
            if np.all(fc < ref):
                old_F = np.array(F_list[:-1])
                dominated = np.any(np.all(old_F <= fc, axis=1) & np.any(old_F < fc, axis=1))
                if not dominated:
                    step_size = max(0.05, step_size * 0.92)
                else:
                    step_size = min(1.0, step_size * 1.08)
            else:
                step_size = min(1.0, step_size * 1.15)

        return X_list
# EVOLVE-BLOCK-END
