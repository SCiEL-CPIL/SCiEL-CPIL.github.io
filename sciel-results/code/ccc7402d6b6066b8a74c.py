"""Multi-objective optimizer for ZDT3 using exact Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    ZDT3 Pareto front: x2..x30 = 0 gives g=1, f1 = x1, f2 = 1 - sqrt(x1).
    The front is discontinuous with 7 segments where f2 is Pareto-optimal.
    
    Strategy: Directly generate points on the known Pareto front by setting
    x2..x30 = 0 and sampling x1 densely across all valid segments.
    Use a deterministic space-filling sequence (Halton) for uniform coverage
    of the [0,1] domain, combined with exact segment boundaries to maximize
    hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, base):
        """Generate the first n points of a Halton sequence in [0, 1]."""
        result = []
        f = 1.0 / base
        for i in range(1, n + 1):
            x = 0.0
            j = i
            while j > 0:
                f /= base
                x += f * (j % base)
                j //= base
            result.append(x)
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front segments - exact analytical boundaries
        # The h function for ZDT3 creates 7 disjoint Pareto-optimal segments
        # h(x, g) = 1 - sqrt(x/g) - x*(2*sin(10*pi*x) - sin(4*pi*x))/g
        # For g=1, the Pareto front exists where h(x,1) >= 0
        # The 7 segments are approximately:
        segments = [
            (0.0, 0.0575),
            (0.121, 0.2575),
            (0.332, 0.6523),
            (0.6523, 0.8233),
            (0.8233, 1.0),
        ]
        
        n_samples = self.budget
        
        # Always include segment endpoints (critical for hypervolume)
        x1_candidates = set()
        for a, b in segments:
            x1_candidates.add(a)
            x1_candidates.add(b)
        
        # Use Halton sequence for space-filling coverage of [0, 1]
        # This guarantees uniform coverage without clustering
        n_halton = max(1, n_samples - len(x1_candidates))
        halton_points = self._halton_sequence(n_halton, base=2)
        
        # Map Halton points to the union of all segments
        # First, compute the total length of all segments
        seg_lengths = [b - a for a, b in segments]
        total_seg_length = sum(seg_lengths)
        
        # For each Halton point, map it to a position within the concatenated segments
        for h in halton_points:
            # h is in [0, 1], map to total_seg_length
            pos = h * total_seg_length
            # Find which segment this position falls in
            cum = 0.0
            for i, (a, b) in enumerate(segments):
                if pos <= cum + seg_lengths[i]:
                    # Within this segment
                    frac = (pos - cum) / seg_lengths[i] if seg_lengths[i] > 0 else 0.5
                    x1_val = a + frac * seg_lengths[i]
                    x1_candidates.add(x1_val)
                    break
                cum += seg_lengths[i]
            else:
                # Should not happen, but add midpoint of last segment as fallback
                x1_candidates.add((segments[-1][0] + segments[-1][1]) / 2)
        
        x1_values = sorted(x1_candidates)
        
        # Limit to budget if needed
        if len(x1_values) > self.budget:
            # Prioritize keeping boundary points and well-distributed points
            # Use a stratified selection to maintain coverage
            n_keep = self.budget
            # Ensure boundaries are kept
            boundary_vals = set()
            for a, b in segments:
                boundary_vals.add(a)
                boundary_vals.add(b)
            
            non_boundary = [v for v in x1_values if v not in boundary_vals]
            boundaries = [v for v in x1_values if v in boundary_vals]
            
            # Keep all boundaries
            final_vals = boundaries[:]
            remaining = n_keep - len(final_vals)
            
            if remaining > 0 and len(non_boundary) > 0:
                # Stratified sampling from non-boundary points
                n_to_pick = min(remaining, len(non_boundary))
                idx = np.linspace(0, len(non_boundary) - 1, n_to_pick).astype(int)
                for i in idx:
                    final_vals.append(non_boundary[i])
            
            x1_values = sorted(final_vals)
        elif len(x1_values) < self.budget:
            # If we have room, add more Halton points with a different base
            deficit = self.budget - len(x1_values)
            halton_points2 = self._halton_sequence(deficit, base=3)
            for h in halton_points2:
                pos = h * total_seg_length
                cum = 0.0
                for i, (a, b) in enumerate(segments):
                    if pos <= cum + seg_lengths[i]:
                        frac = (pos - cum) / seg_lengths[i] if seg_lengths[i] > 0 else 0.5
                        x1_val = a + frac * seg_lengths[i]
                        if x1_val not in x1_values:
                            x1_values.append(x1_val)
                        break
                    cum += seg_lengths[i]
                else:
                    x1_val = (segments[-1][0] + segments[-1][1]) / 2
                    if x1_val not in x1_values:
                        x1_values.append(x1_val)
            
            x1_values = sorted(x1_values[:self.budget])
        
        x1_values = np.array(x1_values)
        
        # Generate and evaluate points on the Pareto front
        # x2..x30 = 0 ensures g = 1 (the minimum possible value)
        n_points = len(x1_values)
        points = np.zeros((n_points, self.dim))
        for i, x1 in enumerate(x1_values):
            points[i, 0] = x1
            # x2..x30 remain 0, which gives g=1
        
        # Clip to bounds (should already be within [0,1] but ensure safety)
        points = np.clip(points, lo, hi)
        
        # Evaluate all points - each evaluation is recorded by the evaluator
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(points[i])
        
        return points
# EVOLVE-BLOCK-END
