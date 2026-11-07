import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point
        ref_point = np.array([1.1, 1.3, 1.1])
        
        # Generate candidate lattice (LHS)
        n_candidates = 100
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        lhs_samples = sampler.random(n=n_candidates)
        candidates = lo + lhs_samples * (hi - lo)
        
        archive = []
        archive_objs = []
        evaluated_indices = set()
        
        X_train = []
        F_train = []
        
        # Helper: Fast 3D Hypervolume Calculation (O(N^3) but N is small)
        def fast_hv_3d(pts, ref):
            if len(pts) == 0:
                return 0.0
            pts = np.array(pts)
            mask = np.all(pts < ref, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            n = len(pts)
            
            def hv_2d(pts_2d, ref_2d):
                if len(pts_2d) == 0:
                    return 0.0
                pts_2d = np.array(pts_2d)
                mask = np.all(pts_2d < ref_2d, axis=1)
                pts_2d = pts_2d[mask]
                if len(pts_2d) == 0:
                    return 0.0
                idx = np.argsort(pts_2d[:, 0])
                pts_2d = pts_2d[idx]
                n_2d = len(pts_2d)
                area = 0.0
                y_max = ref_2d[1]
                for i in range(n_2d):
                    x_curr = pts_2d[i, 0]
                    y_curr = pts_2d[i, 1]
                    x_next = ref_2d[0] if i == n_2d - 1 else pts_2d[i+1, 0]
                    if y_curr < y_max:
                        area += (x_next - x_curr) * (y_max - y_curr)
                        y_max = y_curr
                return area

            total_vol = 0.0
            for i in range(n):
                x_curr = pts[i, 0]
                x_next = ref[0] if i == n - 1 else pts[i+1, 0]
                width = x_next - x_curr
                prefix_2d = pts[0:i+1, 1:3]
                area = hv_2d(prefix_2d, ref[1:3])
                total_vol += width * area
            return total_vol

        # Helper: KNN Prediction
        def predict_knn(x_new, X_train, F_train, k=10):
            if len(X_train) == 0:
                return np.zeros(self.n_obj)
            
            dists = np.linalg.norm(X_train - x_new, axis=1)
            k = min(k, len(X_train))
            nn_indices = np.argpartition(dists, k)[:k]
            weights = 1.0 / (dists[nn_indices] + 1e-8)
            weights /= np.sum(weights)
            pred = np.dot(weights, F_train[nn_indices])
            return pred

        # Helper: Update Archive
        def update_archive(new_point, new_obj):
            dominated = False
            for af in archive_objs:
                if np.all(af <= new_obj) and np.any(af < new_obj):
                    dominated = True
                    break
            
            if dominated:
                return archive, archive_objs
            
            new_archive = []
            new_archive_objs = []
            for p, f in zip(archive, archive_objs):
                if not (np.all(new_obj <= f) and np.any(new_obj < f)):
                    new_archive.append(p)
                    new_archive_objs.append(f)
            
            new_archive.append(new_point)
            new_archive_objs.append(new_obj)
            return new_archive, new_archive_objs

        # Initialization
        init_count = 10
        rng = np.random.default_rng(42)
        init_indices = rng.choice(n_candidates, size=init_count, replace=False)
        
        for i in init_indices:
            x = candidates[i]
            f = problem(x)
            evaluated_indices.add(i)
            X_train.append(x)
            F_train.append(f)
            archive, archive_objs = update_archive(x, f)
        
        X_train = np.array(X_train)
        F_train = np.array(F_train)
        
        # Main Loop
        for step in range(init_count, self.budget):
            remaining = [i for i in range(n_candidates) if i not in evaluated_indices]
            if not remaining:
                break
            
            current_hv = fast_hv_3d(archive_objs, ref_point)
            
            best_marginal = -1.0
            best_idx = -1
            
            for idx in remaining:
                x = candidates[idx]
                f_pred = predict_knn(x, X_train, F_train)
                
                dominated = False
                for af in archive_objs:
                    if np.all(af <= f_pred) and np.any(af < f_pred):
                        dominated = True
                        break
                if dominated:
                    continue
                
                new_objs = archive_objs + [f_pred]
                new_hv = fast_hv_3d(new_objs, ref_point)
                marginal = new_hv - current_hv
                
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_idx = idx
            
            if best_idx == -1:
                best_idx = remaining[0]
            
            x = candidates[best_idx]
            f = problem(x)
            evaluated_indices.add(best_idx)
            
            X_train = np.vstack([X_train, x])
            F_train = np.vstack([F_train, f])
            
            archive, archive_objs = update_archive(x, f)
        
        return archive
