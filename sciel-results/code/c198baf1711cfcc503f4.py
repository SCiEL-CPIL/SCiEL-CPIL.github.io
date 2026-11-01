"""Simulated Annealing for 50 equal circles in a unit square."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0

    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Step 1: Get a good initial configuration
    cents, r0 = _construct_hexagonal()
    current_cents = cents.copy()
    best_cents = cents.copy()
    best_r = r0
    current_r = r0

    # Simulated Annealing Parameters
    # We want to maximize the minimum distance.
    # Objective: f(x) = min_dist
    # Move: x_new = x_old + delta
    # Acceptance: if f(x_new) > f(x_old), accept.
    # Otherwise, accept with probability exp((f(x_new) - f(x_old)) / T).
    
    # Initial temperature should be high enough to explore, low enough to converge.
    # Distances are in range [0, 0.7]. 
    # Let's start T around 0.05 and cool down.
    T_start = 0.05
    T_end = 1e-6
    alpha = 0.999 # Cooling factor per step
    
    # Step size
    # Initial step size should be comparable to the desired radius change or circle diameter.
    # r ~ 0.07, diameter ~ 0.14.
    # Let's start with step = 0.05.
    step_size = 0.05
    
    # We will run until time runs out.
    # Each iteration: 
    # 1. Pick a random circle index i.
    # 2. Generate a random displacement (dx, dy).
    # 3. Move circle i.
    # 4. Check bounds. If out of bounds, reflect or clamp? 
    #    Clamping is safer for validity but might trap at walls.
    #    Reflection is better for SA.
    # 5. Compute new score.
    # 6. Accept/Reject.
    
    # Optimization: 
    # Computing the full score (O(N^2)) every step is expensive.
    # N=50, N^2=2500 operations. 
    # 120s budget. 
    # Python loop overhead is high.
    # Let's estimate: 10,000 steps/sec is optimistic in pure Python with NumPy overhead.
    # Actually, _score does vectorized ops. 
    # diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :] is (50, 50, 2).
    # dists is (50, 50).
    # This is fast in C via NumPy.
    # Let's try to run as many steps as possible.
    
    # To speed up, we can pre-allocate arrays if possible, but dynamic resizing is hard.
    # We'll just call _score.
    
    # Heuristic: 
    # If we move one circle, only its distances to others and walls change.
    # But _score computes min over all. 
    # If the current min was determined by a pair NOT involving i, and the new min is still determined by that pair, 
    # the score doesn't change. 
    # However, implementing an incremental update is complex and error-prone in Python.
    # Let's stick to full recomputation but keep the loop tight.
    
    T = T_start
    steps = 0
    
    while time.time() - start_time < deadline - 5:
        # Pick random index
        i = np.random.randint(0, N)
        
        # Generate random displacement
        # Use Gaussian noise for smoother exploration? Or uniform?
        # Gaussian is standard for SA.
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_cents = current_cents.copy()
        new_cents[i, 0] += dx
        new_cents[i, 1] += dy
        
        # Handle boundaries
        # If a circle goes out, it reduces the wall distance score significantly.
        # We can just let the score penalize it.
        # But the problem requires centers to be INSIDE. 
        # The score function handles this by returning min(wall, pair).
        # If wall < 0, score is negative? 
        # "min_k (1/2 - |c_ik|)" -> if |c| > 0.5, this is negative.
        # The task says "A centre on or outside the square ... make the candidate invalid."
        # Wait, the score function in the prompt:
        # r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
        # If a point is outside, 1/2 - |c| is negative. 
        # So r becomes negative. 
        # We want to maximize r. 
        # So SA will naturally reject moves that push points outside because r drops.
        # However, we must ensure the FINAL output is valid.
        # The prompt says: "A centre on or outside the square ... make the candidate invalid."
        # So we must clip the final result to [-0.5, 0.5].
        
        new_r = _score(new_cents)
        
        # Acceptance criterion
        if new_r >= current_r:
            current_cents = new_cents
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_cents = current_cents.copy()
        else:
            # Delta = new_r - current_r (negative)
            delta = new_r - current_r
            # Probability of accepting a worse move
            # exp(delta / T)
            # If delta is very negative, prob is 0.
            # To avoid underflow/overflow issues with exp, we can check if delta < -T * 100 (approx)
            # But standard exp is fine for reasonable T.
            prob = np.exp(delta / T)
            if np.random.rand() < prob:
                current_cents = new_cents
                current_r = new_r
        
        # Cool down
        T *= alpha
        
        # Occasionally restart temperature if stuck? 
        # Or just let it cool. 
        # If T becomes very small, we do local search.
        # If we want to restart, we can check if best_r hasn't improved for X steps.
        
        steps += 1
        
        # Simple restart strategy:
        # If we haven't improved best_r for a long time (e.g., 5000 steps),
        # and T is still high, maybe perturb best_cents and restart from there?
        # Or just continue. SA usually works well with a single long run.
        
        # Let's add a "kick" if stuck.
        if steps % 10000 == 0:
            # Check time
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                break
            
            # If we've been stuck for a while, maybe increase step size slightly or reset T?
            # For now, just continue.
            
    # Final polish:
    # The SA might end with points slightly outside due to numerical drift or just bad luck in the last step.
    # We must ensure validity.
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    # Re-check score after clipping
    final_r = _score(best_cents)
    
    # If clipping made it worse, we might want to try a few local moves from the clipped state?
    # Or just return it. The score is what it is.
    
    return best_cents
# EVOLVE-BLOCK-END
