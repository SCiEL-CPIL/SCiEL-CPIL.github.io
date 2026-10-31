import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_sq_dists(pts):
        """Compute squared pairwise distances."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        """
        d_sq = get_sq_dists(pts)
        min_d_sq = np.min(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        max_d_sq = np.max(d_sq)
        # Ratio of distances is sqrt(max_d_sq/min_d_sq)
        # Squared ratio is max_d_sq/min_d_sq
        return max_d_sq / min_d_sq

    def soft_score(pts, beta=50.0):
        """
        Smooth surrogate for the objective function to allow gradient-based optimization.
        Uses log-sum-exp for soft min and max of squared distances.
        """
        d_sq = get_sq_dists(pts)
        d_sq = np.maximum(d_sq, 1e-12)
        
        max_d_sq = np.max(d_sq)
        # Soft max
        soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        min_d_sq = np.min(d_sq)
        # Soft min
        soft_min = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min < 1e-8:
            return 1e10
            
        return soft_max / soft_min

    def normalize(pts):
        """Center and scale points to unit variance for stability."""
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    def particle_equilibrium_seed(rng, n_iters=200):
        """
        Generate seed from particle equilibrium with short-range repulsion.
        This tends to create locally uniform distributions.
        """
        pts = rng.uniform(-1, 1, (N, 2))
        for _ in range(n_iters):
            if time.time() > deadline - 60:
                break
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    diff = pts[i] - pts[j]
                    dist_sq = np.sum(diff**2)
                    dist = np.sqrt(dist_sq) + 1e-10
                    # Repulsion 1/r^4
                    force_mag = 1.0 / (dist_sq**2 + 0.01)
                    force = force_mag * diff / dist
                    forces[i] += force
                    forces[j] -= force
            
            # Boundary pressure
            for i in range(N):
                r = np.linalg.norm(pts[i])
                if r > 1.0:
                    forces[i] -= 0.5 * (r - 1.0) * pts[i] / r
            
            pts += 0.01 * forces
            pts = np.clip(pts, -1.5, 1.5)
        
        return normalize(pts)

    def lattice_torus_seed(rng, aspect_ratio=None):
        """
        Generate seed based on a 4x4 lattice with a specific aspect ratio and jitter.
        This implements the 'discrete torus/lattice' structural idea as a geometric prior.
        """
        if aspect_ratio is None:
            aspect_ratio = rng.uniform(0.8, 1.2)
        
        # 4x4 grid indices
        x = np.arange(4)
        y = np.arange(4)
        xx, yy = np.meshgrid(x, y)
        
        # Center the grid
        xx = xx - 1.5
        yy = yy - 1.5
        
        # Apply aspect ratio to x
        xx = xx * aspect_ratio
        
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Add small random jitter to break symmetry and allow escape
        jitter = rng.normal(0, 0.1, size=(N, 2))
        pts += jitter
        
        return normalize(pts)

    def simulated_annealing(init_pts, max_iters=5000, T_start=0.5, T_end=0.01, alpha=None):
        """
        Simulated Annealing to escape local minima.
        """
        if alpha is None:
            alpha = (T_end / T_start) ** (1.0 / max(1, max_iters - 1))
        
        pts = init_pts.copy()
        current_score = score_raw(pts)
        best_pts = pts.copy()
        best_score = current_score
        T = T_start
        
        rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
        
        for i in range(max_iters):
            if time.time() > deadline - 15:
                break
                
            # Perturb a random point
            idx = rng.integers(0, N)
            displacement = rng.normal(0, 0.1, 2)
            new_pts = pts.copy()
            new_pts[idx] += displacement
            new_pts = normalize(new_pts)
            
            new_score = score_raw(new_pts)
            
            # Accept if better or with probability exp(-delta/T)
            if new_score < current_score or rng.random() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
                pts = new_pts
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            
            T *= alpha
            
        return best_pts, best_score

    def refine_with_soft_objective(pts):
        """
        Use L-BFGS-B on the soft objective to polish the configuration.
        """
        if time.time() > deadline - 10:
            return pts, score_raw(pts)
            
        def soft_obj(x, beta=100.0):
            return soft_score(x.reshape(N, 2), beta)
        
        try:
            res = minimize(soft_obj, pts.ravel(), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            refined_pts = res.x.reshape(N, 2)
            refined_score = score_raw(refined_pts)
            return refined_pts, refined_score
        except Exception:
            return pts, score_raw(pts)

    def final_polish(pts):
        """
        Short Powell optimization on the exact objective for final precision.
        """
        if time.time() > deadline - 5:
            return pts, score_raw(pts)
            
        def exact_obj(x):
            return score_raw(x.reshape(N, 2))
        
        try:
            res = minimize(exact_obj, pts.ravel(), method='Powell',
                           options={'maxiter': 100, 'maxfev': 1000})
            polished_pts = res.x.reshape(N, 2)
            polished_score = score_raw(polished_pts)
            return polished_pts, polished_score
        except Exception:
            return pts, score_raw(pts)

    # --- Main Search Loop ---
    
    best_score = 1e10
    best_pts = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Particle Equilibrium Seeds
    for _ in range(3):
        seeds.append(particle_equilibrium_seed(rng))
        
    # 2. Lattice/Torus Seeds (Structural Change: Lattice-based prior)
    for aspect in [0.8, 1.0, 1.2, 1.5]:
        seeds.append(lattice_torus_seed(rng, aspect_ratio=aspect))
        
    # 3. Concentric Rings (Known good structure for N=16)
    r_in, r_out = 0.5, 1.2
    angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    pts_in = np.column_stack([r_in*np.cos(angles_in), r_in*np.sin(angles_in)])
    pts_out = np.column_stack([r_out*np.cos(angles_out), r_out*np.sin(angles_out)])
    seeds.append(normalize(np.vstack([pts_in, pts_out])))
    
    # 4. Random Seeds
    for _ in range(2):
        seeds.append(normalize(rng.uniform(-1, 1, (N, 2))))

    # Score seeds and sort
    scored_seeds = []
    for s in seeds:
        s_score = score_raw(s)
        scored_seeds.append((s_score, s))
        
    scored_seeds.sort(key=lambda x: x[0])
    
    # Process top seeds with SA + Refinement
    # We have ~120s budget. SA is expensive. Let's process top 5 seeds deeply.
    num_to_process = min(5, len(scored_seeds))
    
    for i in range(num_to_process):
        if time.time() > deadline - 20:
            break
            
        curr_score, pts = scored_seeds[i]
        
        if curr_score < best_score:
            best_score = curr_score
            best_pts = pts.copy()
            
        # Phase 1: Simulated Annealing
        # Adjust iterations based on remaining time
        time_remaining = deadline - time.time()
        sa_iters = int(3000 * (time_remaining / 100.0))
        sa_iters = max(500, min(5000, sa_iters))
        
        sa_pts, sa_score = simulated_annealing(pts, max_iters=sa_iters)
        
        if sa_score < best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
            
        # Phase 2: Soft Objective Refinement
        refined_pts, refined_score = refine_with_soft_objective(sa_pts)
        
        if refined_score < best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
            
        # Phase 3: Final Polish
        polished_pts, polished_score = final_polish(refined_pts)
        
        if polished_score < best_score:
            best_score = polished_score
            best_pts = polished_pts.copy()
            
        # If we found a very good score, we can stop early or do one more local restart
        if best_score < 1.35:
            break

    # Fallback if no best_pts found (should not happen)
    if best_pts is None:
        best_pts = scored_seeds[0][1]
        
    # Center the final points
    best_pts = best_pts - np.mean(best_pts, axis=0)
    
    return best_pts
