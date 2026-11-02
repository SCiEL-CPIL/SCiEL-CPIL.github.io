import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    best_score = -np.inf
    best_result = None
    
    # We will optimize over aspect ratios as well.
    # Perimeter is 4, so 2(w+h) = 4 => w+h=2.
    # w in (0, 2). h = 2 - w.
    
    def evaluate_and_fix(x, y, r, w, h):
        """
        Calculates the violation score and fixes positions if they are deeply invalid.
        Returns (violation, sum_r, valid_bool).
        """
        violation = 0.0
        sum_r = np.sum(r)
        valid = True
        
        # Boundary violations
        # We want x in [r, w-r] and y in [r, h-r]
        # Violation = max(0, r - x) + max(0, x - (w-r)) + ...
        
        # Efficient vectorized check for boundaries
        # Left: x < r  => viol = r - x
        # Right: x > w - r => viol = x - (w - r)
        # Bottom: y < r => viol = r - y
        # Top: y > h - r => viol = y - (h - r)
        
        viol_left = np.maximum(0, r - x)
        viol_right = np.maximum(0, x - (w - r))
        viol_bottom = np.maximum(0, r - y)
        viol_top = np.maximum(0, y - (h - r))
        
        boundary_viol = np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top)
        
        if boundary_viol > 1e-6:
            valid = False
        violation += boundary_viol * 10.0  # Weight boundary heavily
        
        # Pairwise overlap violations
        # dist_ij < r_i + r_j => viol = r_i + r_j - dist_ij
        # We can vectorize this using broadcasting, but for N=21, a loop or sparse approach is fine.
        # Broadcasting: (N, 1, 1) vs (1, N, 1) etc.
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Overlap violation is max(0, r_sum - dist)
        # Diagonal is 0, so ignore.
        overlap_viol = np.maximum(0, r_sum - dist)
        # Sum only upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        total_overlap = np.sum(overlap_viol[iu])
        
        if total_overlap > 1e-6:
            valid = False
        violation += total_overlap * 10.0
        
        return violation, sum_r, valid

    def force_directed_optimize(w, h, seed_val):
        """
        Uses a force-directed (gradient descent with momentum/annealing) approach.
        """
        rng = np.random.default_rng(seed_val)
        
        # Initialization: Random positions within a safe margin, small radii
        # Or grid-based initialization for better start
        # Let's try a few initialization strategies
        best_local_score = -np.inf
        best_local_x = None
        best_local_y = None
        best_local_r = None
        
        init_strategies = 5
        
        for init_iter in range(init_strategies):
            # Initialize positions
            if init_iter == 0:
                # Grid-like
                cols = int(np.ceil(np.sqrt(N * w / h)))
                rows = int(np.ceil(N / cols))
                xs = []
                ys = []
                for i in range(cols):
                    for j in range(rows):
                        if len(xs) >= N: continue
                        xs.append((i + 0.5) * w / cols + rng.uniform(-0.02, 0.02))
                        ys.append((j + 0.5) * h / rows + rng.uniform(-0.02, 0.02))
                x = np.array(xs[:N])
                y = np.array(ys[:N])
                r = np.full(N, 0.05)
            else:
                # Random
                x = rng.uniform(0.1, w - 0.1, N)
                y = rng.uniform(0.1, h - 0.1, N)
                r = rng.uniform(0.02, 0.08, N)
            
            # Ensure initial state is not catastrophically bad? 
            # The optimizer will handle it, but if violation is huge, it might diverge.
            # Let's clamp initial positions to be inside.
            x = np.clip(x, 0.01, w - 0.01)
            y = np.clip(y, 0.01, h - 0.01)
            
            current_x = x.copy()
            current_y = y.copy()
            current_r = r.copy()
            
            # Momentum
            vx = np.zeros(N)
            vy = np.zeros(N)
            vr = np.zeros(N)
            
            T = 1.0  # Temperature for annealing
            T_min = 0.01
            alpha = 0.995  # Cooling rate
            
            # Optimization loop
            max_iters = 2000
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    return best_local_score, best_local_x, best_local_y, best_local_r
                
                # Calculate forces (gradients of violation + objective)
                # Objective: Maximize sum(r). So gradient is -1 for r.
                # Violation: Minimize V. Gradient is dV/dx, dV/dy, dV/dr.
                
                # We compute forces as: F = -grad(V) + lambda * grad(Objective)
                # Actually, we want to move in direction of decreasing V and increasing sum_r.
                # Let's define a potential energy E = V - lambda * sum_r
                # We want to minimize E.
                # lambda controls how much we care about size vs validity.
                # We can adapt lambda. Start small, increase as V decreases?
                # Or just use a fixed penalty method.
                
                # Let's compute gradients explicitly.
                fx = np.zeros(N)
                fy = np.zeros(N)
                fr = np.ones(N) # Gradient of -sum(r) is -1, so force is +1 (increase r)
                
                # Boundary gradients
                # V_bound = sum(max(0, r-x) + ...)
                # dV/dx for left: -1 if x < r else 0
                # dV/dx for right: 1 if x > w-r else 0
                # dV/dr for left: 1 if x < r else 0
                # dV/dr for right: -1 if x > w-r else 0 (since w-r decreases as r increases? No, term is x-(w-r) = x-w+r. d/dr is 1)
                
                # Let's re-evaluate boundary terms carefully.
                # Term 1: max(0, r - x). 
                #   if active: d/dx = -1, d/dr = 1
                # Term 2: max(0, x - (w - r)) = max(0, x - w + r).
                #   if active: d/dx = 1, d/dr = 1
                # Term 3: max(0, r - y).
                #   if active: d/dy = -1, d/dr = 1
                # Term 4: max(0, y - (h - r)) = max(0, y - h + r).
                #   if active: d/dy = 1, d/dr = 1
                
                active_left = (current_x < current_r)
                active_right = (current_x > w - current_r)
                active_bottom = (current_y < current_r)
                active_top = (current_y > h - current_r)
                
                fx += -active_left * 10.0
                fx += active_right * 10.0
                fy += -active_bottom * 10.0
                fy += active_top * 10.0
                fr += (active_left + active_right + active_bottom + active_top) * 10.0
                
                # Pairwise overlap gradients
                # V_overlap = sum_{i<j} max(0, r_i + r_j - dist_ij)
                # Let d_ij be distance.
                # Term: max(0, r_i + r_j - d_ij)
                # If active:
                #   d/dx_i = (x_i - x_j) / d_ij  (from -d_ij derivative: - (x_i-x_j)/d_ij * -1? No.)
                #   dist = sqrt((xi-xj)^2 + ...). d(dist)/dxi = (xi-xj)/dist.
                #   d(r_i + r_j - dist)/dxi = - (xi-xj)/dist.
                #   So force on i is - grad(V) = + (xi-xj)/dist. This pushes i away from j. Correct.
                
                dx = current_x[:, np.newaxis] - current_x[np.newaxis, :]
                dy = current_y[:, np.newaxis] - current_y[np.newaxis, :]
                dist = np.sqrt(dx**2 + dy**2)
                
                # Avoid division by zero
                dist[dist < 1e-8] = 1e-8
                
                r_sum = current_r[:, np.newaxis] + current_r[np.newaxis, :]
                
                # Active overlaps: r_sum > dist
                active_overlap = (r_sum > dist)
                
                # Gradient of V_overlap w.r.t x_i:
                # sum_j (x_i - x_j)/dist_ij * I(active_ij)
                # Note: This is sum over j != i.
                # The matrix A_ij = (x_i - x_j)/dist_ij * I(active_ij)
                # Force on i is - sum_j A_ij (from V) + ... 
                # Wait, V = sum max(0, r_i+r_j - d).
                # dV/dx_i = sum_{j != i} (x_i - x_j)/d_ij * I(active)
                # Force F_i = - dV/dx_i = - sum_{j != i} (x_i - x_j)/d_ij * I(active)
                # This points from i towards j?
                # x_i - x_j is vector from j to i.
                # If i is to the right of j, x_i > x_j, vector is positive.
                # Force is negative, so pushes i to the left (towards j)?
                # No, we want to push them apart.
                # Let's check: If x_i > x_j, they are overlapping. We want to increase x_i or decrease x_j.
                # Gradient of (r_i+r_j - dist) w.r.t x_i is -(x_i-x_j)/dist.
                # Since x_i > x_j, x_i-x_j > 0, so gradient is negative.
                # We minimize V, so we move in direction -grad(V).
                # -grad(V) = (x_i-x_j)/dist. Positive. So x_i increases. Correct.
                
                # So Force_x_i = sum_{j != i} (x_i - x_j)/dist_ij * I(active_ij)
                
                # We can compute this via matrix multiplication.
                # M_ij = (x_i - x_j)/dist_ij if active else 0.
                # F_i = sum_j M_ij.
                
                M_x = np.where(active_overlap, dx / dist, 0)
                M_y = np.where(active_overlap, dy / dist, 0)
                
                # Sum over j (axis 1)
                fx += np.sum(M_x, axis=1) * 10.0
                fy += np.sum(M_y, axis=1) * 10.0
                
                # Gradient w.r.t r_i
                # dV/dr_i = sum_{j != i} I(active_ij)
                # Force on r_i = - sum I(active) + 1 (from objective)
                fr -= np.sum(active_overlap, axis=1) * 10.0
                
                # Objective gradient: Maximize sum(r).
                # In minimization of E = V - lambda*sum_r, grad_r = -lambda.
                # We set fr = 1 initially (unit force to grow).
                # Let's scale the growth force based on temperature or just keep it constant.
                # Actually, the "1" in fr above is the force to grow.
                # The penalty forces are scaled by 10.
                # This balance determines the final size.
                
                # Update velocities with momentum and damping
                vx = 0.9 * vx + 0.1 * fx
                vy = 0.9 * vy + 0.1 * fy
                vr = 0.9 * vr + 0.1 * fr
                
                # Step size (learning rate)
                # Anneal the step size
                step = 0.05 * T
                
                current_x += step * vx
                current_y += step * vy
                current_r += step * vr
                
                # Clamp positions to prevent flying out
                current_x = np.clip(current_x, 0.0, w)
                current_y = np.clip(current_y, 0.0, h)
                current_r = np.clip(current_r, 0.001, min(w, h)/2)
                
                # Cool down
                T = max(T_min, T * alpha)
                
                # Check for best
                # Only update best if valid? Or track best valid score.
                # We can check validity periodically to save time.
                if it % 10 == 0:
                    v, s, valid = evaluate_and_fix(current_x, current_y, current_r, w, h)
                    if valid and s > best_local_score:
                        best_local_score = s
                        best_local_x = current_x.copy()
                        best_local_y = current_y.copy()
                        best_local_r = current_r.copy()
                        
                    # If we have a very good valid solution, we can stop early?
                    # Or just let it run.
                    
        return best_local_score, best_local_x, best_local_y, best_local_r

    # Main Loop: Iterate over aspect ratios and seeds
    # We have ~140s.
    # Let's try 10 aspect ratios.
    # For each, run 3-5 seeds.
    
    aspect_ratios = np.linspace(0.5, 1.5, 10)
    
    # Shuffle aspects to ensure we don't run out of time on bad ones?
    # No, deterministic is better for reproducibility, but we want to find the best.
    # Let's just iterate.
    
    num_aspects = len(aspect_ratios)
    seeds_per_aspect = 3
    
    total_runs = num_aspects * seeds_per_aspect
    # Estimated time per run: 140 / 30 ~ 4.6s.
    # Our inner loop is 2000 iters. Vectorized ops on 21x21 matrices are fast.
    # Should be well within budget.
    
    for k, w in enumerate(aspect_ratios):
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            continue
            
        for s in range(seeds_per_aspect):
            if time.time() - start_time > time_limit - 10.0:
                break
                
            seed_val = 42 + k * 100 + s * 7
            
            score, rx, ry, rr = force_directed_optimize(w, h, seed_val)
            
            if rx is not None and score > best_score:
                best_score = score
                best_result = np.column_stack([rx, ry, rr])
                
    if best_result is None:
        # Fallback: Simple grid
        w = 1.0
        h = 1.0
        xs = []
        ys = []
        r = []
        for i in range(5):
            for j in range(5):
                if len(xs) >= N:
                    break
                xs.append((i + 0.5) * 0.2)
                ys.append((j + 0.5) * 0.2)
                r.append(0.1 - 1e-3)
        best_result = np.column_stack([xs, ys, r])
        
    return best_result

# Test/Entry Point
if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print("Sum of radii:", np.sum(res[:, 2]))
