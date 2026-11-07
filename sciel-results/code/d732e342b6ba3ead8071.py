import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 320.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        # Halton sequence for good initial coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim, start=1):
            seq = np.zeros((n, dim))
            for i in range(n):
                for d in range(dim):
                    seq[i, d] = halton(start + i, d + 2)
            return seq

        # Initial phase: uniform sampling with Halton
        n_init = 40
        init_seq = halton_sequence(n_init, dim)
        init_X = lo + init_seq * (hi - lo)

        archive_X = []
        archive_F = []

        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = init_X[i]
            f = evaluate(x)
            if f is not None:
                archive_X.append(x)
                archive_F.append(f)

        # Remove dominated points from archive
        def clean_archive(archive_X, archive_F):
            new_X = []
            new_F = []
            for i in range(len(archive_F)):
                dominated = False
                for j in range(len(archive_F)):
                    if i == j:
                        continue
                    if np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        dominated = True
                        break
                if not dominated:
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            return new_X, new_F

        archive_X, archive_F = clean_archive(archive_X, archive_F)

        # Main optimization loop
        # Use a combination of:
        # 1. Local search around archive points
        # 2. Global exploration with decreasing frequency
        # 3. Adaptive step size based on improvement rate

        step_base = 0.15
        min_step = 0.005
        step_multiplier = step_base

        no_improve_count = 0
        max_no_improve = 15

        while used < budget and time.time() < deadline:
            if not archive_X:
                x = lo + (hi - lo) * np.random.random(dim)
                f = evaluate(x)
                if f is not None:
                    archive_X.append(x)
                    archive_F.append(f)
                continue

            # Decide: local search or global exploration
            # More likely to do local search as we progress
            if used < budget * 0.3:
                do_local = np.random.random() < 0.7
            elif used < budget * 0.7:
                do_local = np.random.random() < 0.85
            else:
                do_local = np.random.random() < 0.95

            if do_local and len(archive_X) > 0:
                # Pick parent from archive
                # Weighted selection: prefer points with better hypervolume contribution
                # For simplicity, pick randomly but with slight preference for extreme points
                idx = np.random.randint(len(archive_X))
                parent = archive_X[idx]

                # Generate child with adaptive step
                step_size = step_multiplier * (hi - lo)
                noise = np.random.randn(dim) * step_size
                child = parent + noise
                child = np.clip(child, lo, hi)
            else:
                # Global exploration
                # Use a mix of uniform random and Halton-like sampling
                if np.random.random() < 0.5:
                    child = lo + (hi - lo) * np.random.random(dim)
                else:
                    # Sample near the boundaries of the current best region
                    # Find the range of archive
                    min_f = np.min(archive_F, axis=0)
                    max_f = np.max(archive_F, axis=0)
                    # This doesn't directly give us x-range, so just use uniform
                    child = lo + (hi - lo) * np.random.random(dim)

            f = evaluate(child)
            if f is None:
                break

            # Check if non-dominated
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break

            if not dominated:
                # Add to archive and remove dominated points
                new_X = []
                new_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not (np.all(f <= af) and np.any(f < af)):
                        new_X.append(ax)
                        new_F.append(af)
                new_X.append(child)
                new_F.append(f)

                archive_X = new_X
                archive_F = new_F

                # Update step size: we made progress
                no_improve_count = 0
                # Optionally increase step slightly to explore more
                # But keep it bounded
                if step_multiplier < step_base * 1.5:
                    step_multiplier *= 1.05
            else:
                no_improve_count += 1
                # Shrink step size if no improvement
                if no_improve_count >= 5:
                    step_multiplier = max(min_step / (hi - lo).max(), step_multiplier * 0.8)

                # If stuck for too long, do a larger jump or restart locally
                if no_improve_count >= max_no_improve:
                    # Reset and do a bigger jump
                    step_multiplier = step_base
                    no_improve_count = 0
                    # Force a global exploration
                    child = lo + (hi - lo) * np.random.random(dim)
                    f2 = evaluate(child)
                    if f2 is not None:
                        # Check if non-dominated
                        dominated2 = False
                        for af in archive_F:
                            if np.all(af <= f2) and np.any(af < f2):
                                dominated2 = True
                                break
                        if not dominated2:
                            new_X = []
                            new_F = []
                            for ax, af in zip(archive_X, archive_F):
                                if not (np.all(f2 <= af) and np.any(f2 < af)):
                                    new_X.append(ax)
                                    new_F.append(af)
                            new_X.append(child)
                            new_F.append(f2)
                            archive_X = new_X
                            archive_F = new_F

        # Final cleanup
        archive_X, archive_F = clean_archive(archive_X, archive_F)

        return all_X
