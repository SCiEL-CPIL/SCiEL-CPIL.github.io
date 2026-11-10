import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # PSO Parameters
        pop_size = 40
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        v_max = 0.2 * (hi - lo) # Max velocity as fraction of range
        archive_size = 100
        
        # Initialize Particles
        X = np.random.rand(pop_size, n) * (hi - lo) + lo
        V = np.zeros((pop_size, n))
        
        # Evaluate Initial Population
        F = np.array([problem(x) for x in X])
        self.eval_count += pop_size
        
        # Initialize Archive and Personal Bests
        archive = []
        pbest_X = X.copy()
        pbest_F = F.copy()
        
        # Add initial solutions to archive
        for i in range(pop_size):
            if self._is_non_dominated(F[i], [a[1] for a in archive]):
                archive.append((X[i], F[i]))
                
        # Limit archive size
        if len(archive) > archive_size:
            archive = self._reduce_archive(archive, archive_size)

        # Main Loop
        while self.eval_count < self.budget:
            # Update velocities and positions
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Select leader from archive using grid
                leader = self._select_leader(archive)
                
                # Update velocity
                r1 = np.random.rand(n)
                r2 = np.random.rand(n)
                V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (leader[0] - X[i])
                
                # Clamp velocity
                V[i] = np.clip(V[i], -v_max, v_max)
                
                # Update position
                X[i] = X[i] + V[i]
                
                # Clip to bounds
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                f_new = problem(X[i])
                self.eval_count += 1
                
                # Update personal best
                if self._is_better(f_new, pbest_F[i]):
                    pbest_X[i] = X[i].copy()
                    pbest_F[i] = f_new.copy()
                    
                # Update archive
                self._update_archive(archive, (X[i].copy(), f_new.copy()), archive_size)
                
            # Periodic archive reduction to maintain diversity
            if len(archive) > archive_size:
                archive = self._reduce_archive(archive, archive_size)

        return X

    def _is_non_dominated(self, f, archive_Fs):
        """Check if f is non-dominated with respect to list of F vectors."""
        for af in archive_Fs:
            if np.all(af <= f) and np.any(af < f):
                return False
        return True

    def _is_better(self, f1, f2):
        """Check if f1 dominates f2."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _update_archive(self, archive, new_point, max_size):
        """Add new point to archive if it is non-dominated, removing dominated points."""
        x, f = new_point
        new_archive = []
        dominated_new = False
        
        for ax, af in archive:
            # Check if new point dominates archive point
            if np.all(f <= af) and np.any(f < af):
                continue # Remove dominated archive point
            # Check if archive point dominates new point
            elif np.all(af <= f) and np.any(af < f):
                dominated_new = True
                new_archive.append((ax, af))
            else:
                new_archive.append((ax, af))
                
        if not dominated_new:
            new_archive.append((x, f))
            
        # Limit size
        if len(new_archive) > max_size:
            new_archive = self._reduce_archive(new_archive, max_size)
            
        archive.clear()
        archive.extend(new_archive)

    def _reduce_archive(self, archive, max_size):
        """Reduce archive size using grid-based selection."""
        if len(archive) <= max_size:
            return archive
            
        # Simple grid-based reduction
        # Normalize objectives
        Fs = np.array([f for _, f in archive])
        min_F = Fs.min(axis=0)
        max_F = Fs.max(axis=0)
        ranges = max_F - min_F
        ranges[ranges == 0] = 1.0
        
        normalized = (Fs - min_F) / ranges
        
        # Create grid
        grid_size = 2 ** self.n_obj
        cell_counts = np.zeros(grid_size, dtype=int)
        
        # Assign to cells
        indices = []
        for i, norm in enumerate(normalized):
            cell_idx = 0
            for k in range(self.n_obj):
                cell_idx = cell_idx * 2 + int(norm[k] >= 0.5)
            indices.append(cell_idx)
            
        # Count
        for idx in indices:
            cell_counts[idx] += 1
            
        # Select points from least crowded cells
        selected_indices = set()
        while len(selected_indices) < max_size:
            # Find cell with min count
            min_count = np.min(cell_counts)
            candidates = [i for i, c in enumerate(cell_counts) if c == min_count]
            if not candidates:
                break
            # Pick random candidate
            chosen_cell = np.random.choice(candidates)
            
            # Find an unselected point in this cell
            found = False
            for i, idx in enumerate(indices):
                if idx == chosen_cell and i not in selected_indices:
                    selected_indices.add(i)
                    found = True
                    break
            if not found:
                break
                
        # If we didn't select enough, just take the first max_size
        if len(selected_indices) < max_size:
            selected_indices = set(range(max_size))
            
        return [archive[i] for i in selected_indices]

    def _select_leader(self, archive):
        """Select a leader from the archive using grid-based selection."""
        if not archive:
            return None
            
        if len(archive) == 1:
            return archive[0]
            
        Fs = np.array([f for _, f in archive])
        min_F = Fs.min(axis=0)
        max_F = Fs.max(axis=0)
        ranges = max_F - min_F
        ranges[ranges == 0] = 1.0
        
        normalized = (Fs - min_F) / ranges
        
        # Create grid
        grid_size = 2 ** self.n_obj
        cell_counts = np.zeros(grid_size, dtype=int)
        
        # Assign to cells
        indices = []
        for i, norm in enumerate(normalized):
            cell_idx = 0
            for k in range(self.n_obj):
                cell_idx = cell_idx * 2 + int(norm[k] >= 0.5)
            indices.append(cell_idx)
            
        # Count
        for idx in indices:
            cell_counts[idx] += 1
            
        # Find cell with minimum count
        min_count = np.min(cell_counts)
        candidates = [i for i, c in enumerate(cell_counts) if c == min_count]
        
        # Pick a random point from the candidates' cells
        chosen_cell = np.random.choice(candidates)
        for i, idx in enumerate(indices):
            if idx == chosen_cell:
                return archive[i]
                
        return archive[0]
