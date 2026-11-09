import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    def compute_radius(pts):
        """Compute the minimum radius for a given set of points."""
        # Distance to walls
        # pts is (N, 2)
        # Walls are at x = +/- 0.5, y = +/- 0.5
        # Distance to wall for point i is min(0.5 - x_i, 0.5 - y_i, 0.5 + x_i, 0.5 + y_i)
        # Actually, since pts are in [-0.5, 0.5], distance to wall is:
        # min(0.5 - pts[:,0], 0.5 - pts[:,1], 0.5 + pts[:,0], 0.5 + pts[:,1])
        # But we need min over all points and all walls.
        
        # Let's check if any point is outside.
        # If outside, radius is negative/invalid.
        
        # Efficient distance to boundary
        # d_x = np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0])
        # d_y = np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
        # d_wall = np.minimum(d_x, d_y)
        # min_wall = np.min(d_wall)
        
        # Pairwise distances
        # Use cdist or manual
        # Manual vectorized
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Wall distances
        # x distances to walls: 0.5 - x, 0.5 + x
        dx = np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0])
        dy = np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
        d_wall = np.minimum(dx, dy)
        min_wall = np.min(d_wall)
        
        # If any point is outside, min_wall or min_pair might be negative?
        # If a point is at x=0.6, dx = min(-0.1, 1.1) = -0.1.
        # So min_wall will be negative.
        # This correctly penalizes invalid configurations.
        
        return min(min_wall, min_pair) / 2.0

    def smooth_neg_radius(pts, beta):
        """
        Compute a smooth approximation of -radius.
        We want to maximize r = min(dists, wall_dists)/2.
        So we minimize -r = -min(...)/2.
        min(a, b) can be approximated by soft-min.
        soft_min(a, b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
        Here we have many terms.
        Let D be the set of all pairwise distances and wall distances.
        r = min(D) / 2.
        -r = -min(D)/2.
        We want to minimize f(pts) = -soft_min(D, beta) / 2.
        
        However, computing all pairwise distances is O(N^2) = 40000 ops.
        This is fast enough for Python/NumPy if we vectorize.
        
        To avoid overflow in exp, we use the log-sum-exp trick.
        soft_min(D, beta) = -1/beta * logsumexp(-beta * D)
        
        So f(pts) = 1/(2*beta) * logsumexp(-beta * D)
        
        We need the gradient of this function.
        We can compute it via finite differences, but that's slow (2N+2 evaluations).
        Alternatively, we can derive the gradient analytically.
        
        Let L = logsumexp(-beta * D) = log(sum_k exp(-beta * d_k))
        f = L / (2 * beta)
        
        df/dx_i = (1/(2*beta)) * dL/dx_i
        dL/dx_i = (1/sum_k exp(-beta d_k)) * sum_k (-beta * exp(-beta d_k) * dd_k/dx_i)
        Let p_k = exp(-beta d_k) / sum_j exp(-beta d_j)  (softmax weights)
        dL/dx_i = sum_k (-beta * p_k * dd_k/dx_i)
        
        So df/dx_i = -0.5 * sum_k (p_k * dd_k/dx_i)
        
        We need dd_k/dx_i for all pairs k and points i.
        This is complex to vectorize directly for the gradient sum.
        
        Alternative: Use a simpler objective that is easier to optimize.
        Maximize the minimum distance is equivalent to minimizing the maximum overlap.
        Or, use a penalty method.
        
        Let's stick to the smooth min but compute gradient via finite differences for simplicity?
        No, 200 points * 2 dims = 400 params. 400 evaluations of O(N^2) function per step.
        400 * 40000 = 16,000,000 ops per gradient step.
        1000 steps = 16e9 ops. Too slow in Python.
        
        We need an analytic gradient or a cheaper objective.
        
        Let's use the analytic gradient for the soft-min.
        
        Steps:
        1. Compute all pairwise distances D (N, N) and wall distances W (N,).
        2. Flatten D (excluding diagonal) and W into a single array V of length M.
           M = N*(N-1) + 2*N (approx, actually pairwise is N*(N-1)/2 unique, but distance matrix is symmetric. 
           Let's use unique pairs to save memory/time? 
           For gradient, we need to map back to coordinates.
           Using full NxN matrix is easier for vectorization but wastes half.
           Let's use full NxN for simplicity in vectorization, ignoring diagonal.
           
           Actually, let's just compute the gradient for the full distance matrix.
           
        3. Compute softmax weights p = exp(-beta * V) / sum(exp(-beta * V)).
           Note: V contains distances. We want to minimize -min(V).
           So we are effectively minimizing a soft-min of V.
           
        4. Compute gradient of V with respect to x.
           V consists of:
           - Pairwise distances d_ij = ||x_i - x_j||
           - Wall distances w_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
             This is non-differentiable at the center. But we can approximate or handle carefully.
             Actually, wall distance is just distance to boundary.
             d_wall_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i).
             This is a min of 4 linear functions.
             The gradient is the gradient of the active function.
             In the soft-min, we sum over all "constraints".
             So we should treat each wall constraint separately in the soft-min sum.
             
             So V should include:
             - d_ij for i < j
             - 0.5 - x_i
             - 0.5 + x_i
             - 0.5 - y_i
             - 0.5 + y_i
             
             Total terms M = N*(N-1)/2 + 4*N.
             
        5. Gradient accumulation:
           grad_x = zeros((N, 2))
           
           For each term k in V:
           grad_x += p[k] * grad(V[k])
           
           Since we want to minimize f = -soft_min(V), the gradient direction for ascent is -grad(f).
           Wait, we are minimizing f. So we do x -= lr * grad(f).
           f = -1/beta * logsumexp(-beta V)
           grad f = -1/beta * (1/sum) * sum(-beta exp(-beta V_k) grad V_k)
                  = sum (p_k grad V_k)
           where p_k = exp(-beta V_k) / sum(exp(-beta V_j))
           
           So grad f = sum_k p_k grad V_k.
           
           We update x = x - lr * grad f.
           
        Let's implement this.
        """
        pts = pts.reshape(-1, 2)
        
        # Compute pairwise distances
        # To save memory and time, we can compute the gradient directly without storing full NxN matrix if we are clever,
        # but for N=200, NxN is 40k floats, which is small.
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Wall distances
        # 4 walls per point
        w1 = 0.5 - pts[:, 0]
        w2 = 0.5 + pts[:, 0]
        w3 = 0.5 - pts[:, 1]
        w4 = 0.5 + pts[:, 1]
        
        # Concatenate all distances into one vector V
        # Pairwise: take upper triangle
        iu = np.triu_indices(N, k=1)
        d_pairs = dists[iu]
        
        w_all = np.concatenate([w1, w2, w3, w4])
        
        V = np.concatenate([d_pairs, w_all])
        
        # Compute softmax weights
        # Shift for stability
        V_shifted = V - np.min(V)
        exp_V = np.exp(-beta * V_shifted)
        sum_exp = np.sum(exp_V)
        p = exp_V / sum_exp
        
        # Compute gradient of V with respect to x
        grad_x = np.zeros_like(pts)
        
        # 1. Pairwise contributions
        # d_ij = ||x_i - x_j||
        # grad d_ij / dx_i = (x_i - x_j) / d_ij
        # grad d_ij / dx_j = (x_j - x_i) / d_ij = - grad d_ij / dx_i
        
        # We have p for each pair.
        # We need to sum p_k * grad(V_k)
        
        # Vectorized approach for pairwise:
        # Create a weight matrix W where W[i,j] = p_k for the pair (i,j)
        # But we only have upper triangle p.
        
        # Let's construct the full gradient contribution.
        # For each pair (i, j) with i < j:
        # grad_i += p_k * (x_i - x_j) / d_ij
        # grad_j += p_k * (x_j - x_i) / d_ij
        
        # We can use broadcasting if we map p to the matrix.
        # p is length N*(N-1)/2.
        
        # Let's just loop over the active pairs? No, 20k pairs.
        # Vectorized:
        # diff[i,j] = x_i - x_j
        # dists[i,j] = d_ij
        
        # We need to add p_k * (diff[i,j] / dists[i,j]) to grad_x[i]
        # and p_k * (diff[j,i] / dists[j,i]) to grad_x[j]
        
        # Let's create a matrix P of shape (N, N) initialized to 0.
        # Fill upper triangle with p.
        P = np.zeros((N, N))
        P[iu] = p[:len(d_pairs)]
        
        # Now, grad_x[i] = sum_j P[i,j] * (x_i - x_j) / d_ij
        # Note: P[i,j] is non-zero only for i < j in the upper triangle, 
        # but the formula sum_j works for all j if we fill the matrix correctly?
        # Wait, p is defined for unique pairs.
        # If we fill P[i,j] for i<j, then for a specific i, we sum over j>i.
        # We also need the contribution from pairs where i > j.
        # For pair (j, i) with j < i, the weight is P[j,i].
        # The gradient contribution to x_i from pair (j,i) is P[j,i] * (x_i - x_j) / d_ij.
        # Note that (x_i - x_j) = - (x_j - x_i).
        # And d_ij = d_ji.
        
        # Let's just compute the full matrix of gradients.
        # G[i,j] = P_full[i,j] * (x_i - x_j) / d_ij
        # where P_full[i,j] is the weight for the pair (i,j).
        # If i < j, weight is p_k.
        # If i > j, weight is p_k for the pair (j,i).
        # So P_full is symmetric?
        # Yes, the weight for the pair is the same regardless of direction.
        # So P_full = P + P.T (but careful with diagonal, which is 0).
        
        P_full = P + P.T
        
        # Avoid division by zero on diagonal
        dists_safe = dists.copy()
        np.fill_diagonal(dists_safe, 1.0)
        
        # grad_x[i] = sum_j P_full[i,j] * (x_i - x_j) / dists_safe[i,j]
        # diff is (N, N, 2)
        # We want to sum over axis 1.
        
        # term = P_full[:, :, np.newaxis] * (diff / dists_safe[:, :, np.newaxis])
        # grad_x = np.sum(term, axis=1)
        
        # This creates a (N, N, 2) array. 200*200*2 = 80k floats. Very small.
        term = P_full[:, :, np.newaxis] * (diff / dists_safe[:, :, np.newaxis])
        grad_x += np.sum(term, axis=1)
        
        # 2. Wall contributions
        # w1 = 0.5 - x_i. grad = -1 for x_i.
        # w2 = 0.5 + x_i. grad = +1 for x_i.
        # w3 = 0.5 - y_i. grad = -1 for y_i.
        # w4 = 0.5 + y_i. grad = +1 for y_i.
        
        # p for walls is the second part of p
        p_walls = p[len(d_pairs):]
        
        # p_w1 = p_walls[0:N]
        # p_w2 = p_walls[N:2N]
        # p_w3 = p_walls[2N:3N]
        # p_w4 = p_walls[3N:4N]
        
        p_w1, p_w2, p_w3, p_w4 = np.split(p_walls, 4)
        
        grad_x[:, 0] += -p_w1 + p_w2
        grad_x[:, 1] += -p_w3 + p_w4
        
        # f = -1/beta * logsumexp(-beta V)
        # We computed grad f = sum p_k grad V_k.
        # Wait, let's re-verify the sign.
        # f = -1/beta * L
        # dL/dx = sum p_k dV_k/dx
        # df/dx = -1/beta * dL/dx = -1/beta * sum p_k dV_k/dx
        # So grad_f = -1/beta * grad_V_weighted
        
        # We want to minimize f.
        # Update: x = x - lr * grad_f
        # x = x - lr * (-1/beta * grad_V_weighted)
        # x = x + lr/beta * grad_V_weighted
        
        # So we add the weighted gradient of V.
        
        return grad_x, -1.0 / (2.0 * beta) * np.log(sum_exp) + np.min(V) / beta # Approx value, not strictly needed if we track best
    
    def optimize(pts_init, time_limit):
        pts = pts_init.copy()
        lr = 0.01
        beta = 100.0 # Start with moderate beta, increase over time
        momentum = 0.9
        v = np.zeros_like(pts)
        
        best_pts = pts.copy()
        best_r = compute_radius(pts)
        
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            # Adaptive beta: increase beta to make soft-min closer to hard-min
            # beta *= 1.01
            # But if beta is too high, gradients become sparse/steep.
            # Let's keep beta constant or increase slowly.
            
            grad_x, _ = smooth_neg_radius(pts, beta)
            
            # Gradient descent step
            v = momentum * v + lr * grad_x
            pts_new = pts + v # Note: we are doing gradient ascent on radius, which is descent on -radius.
                               # Our grad_x is grad of V (distances).
                               # We want to maximize min(V).
                               # So we move in direction of grad_x.
            
            # Clip to bounds to ensure validity during optimization?
            # If we clip, we might get stuck on boundaries.
            # But the wall terms in V should push points away.
            # However, if a point goes outside, V becomes negative, 
            # and exp(-beta * V) becomes huge, dominating the sum.
            # This creates a strong repulsive force from outside.
            # So we don't strictly need clipping, but it's safer.
            
            # Check radius
            r = compute_radius(pts_new)
            
            if r > best_r:
                best_r = r
                best_pts = pts_new.copy()
                pts = pts_new
            else:
                # If step was bad, reduce learning rate?
                # Simple approach: don't update pts, just keep v?
                # Or revert.
                # Let's just not update pts if it's worse, to avoid diverging.
                # But this might stall.
                # Let's try a line search or just keep lr.
                # For now, if worse, don't move.
                pass
            
            # Periodically re-evaluate best
            # and maybe adjust lr
            
        return best_pts, best_r

    # Initializations
    inits = []
    
    # 1. Hexagonal Grid
    s = 0.075
    hex_points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * s + (j % 2) * s / 2.0
            y = j * np.sqrt(3) / 2.0 * s
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    # Filter to square
    mask = (np.abs(hex_points[:, 0]) <= 0.5) & (np.abs(hex_points[:, 1]) <= 0.5)
    hex_points = hex_points[mask]
    # If we have more than N, take closest to center
    if len(hex_points) > N:
        dists = np.sum(hex_points**2, axis=1)
        idx = np.argsort(dists)[:N]
        hex_points = hex_points[idx]
    else:
        # Pad with random if fewer? Unlikely.
        pass
    inits.append(hex_points)
    
    # 2. Random
    np.random.seed(42)
    rand_pts = np.random.rand(N, 2) - 0.5
    inits.append(rand_pts)
    
    # 3. Square Grid
    m = 15
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    dists = np.sum(grid**2, axis=1)
    idx = np.argsort(dists)[:N]
    inits.append(grid[idx])
    
    best_overall = None
    best_r_overall = -1
    
    # Time budget per init
    # Total 110s. 3 inits. ~36s each.
    # But we can do more restarts if we are fast.
    # Let's do 3 restarts per init? No, just 1 long run per init first.
    # If time allows, do more.
    
    num_inits = len(inits)
    time_per_init = 100.0 / num_inits
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        remaining = deadline - 5 - time.time()
        t_limit = min(time_per_init, remaining)
        
        if t_limit < 5:
            break
            
        # Jitter the init slightly to break symmetry
        init_jittered = init + np.random.normal(0, 0.01, size=init.shape)
        
        pts, r = optimize(init_jittered, t_limit)
        
        if r > best_r_overall:
            best_r_overall = r
            best_overall = pts
            
        # Check time
        if time.time() > deadline - 10:
            break
            
    if best_overall is None:
        # Fallback
        return inits[0]
        
    return best_overall

# Test run
if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
