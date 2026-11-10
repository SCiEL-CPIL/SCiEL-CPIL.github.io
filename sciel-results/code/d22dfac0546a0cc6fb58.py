import numpy as np
import time

N = 50

def construct():
    # Use a fixed seed for reproducibility in evaluation, but maybe random is better for exploration?
    # The task says "The evaluator measures the raw value". It doesn't say we must be deterministic.
    # However, standard practice for these tasks is to be deterministic or use a fixed seed to ensure the score is reproducible.
    # I will use a fixed seed.
    np.random.seed(12345)
    
    def get_score(coords):
        """
        Calculate the maximum radius r for a given configuration.
        coords: (N, 2) array.
        Returns r.
        """
        # Wall distances
        # min(0.5 - x, 0.5 - y, x + 0.5, y + 0.5)
        # Since coords are in [-0.5, 0.5], x+0.5 >= 0, etc.
        # wall_x = 0.5 - np.abs(coords[:, 0])
        # wall_y = 0.5 - np.abs(coords[:, 1])
        # wall_min = np.min(np.minimum(wall_x, wall_y))
        
        # Pairwise distances
        # diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        # dist = np.linalg.norm(diff, axis=2)
        # np.fill_diagonal(dist, np.inf)
        # min_pair = np.min(dist)
        
        # r = min(min_pair / 2.0, wall_min)
        
        # Optimization: Early exit or vectorization is already good.
        # Let's check bounds first to ensure validity? The problem says invalid if outside.
        # We should constrain coordinates to [-0.5, 0.5] during moves.
        
        wall_x = 0.5 - np.abs(coords[:, 0])
        wall_y = 0.5 - np.abs(coords[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        
        # If min_wall < 0, it's invalid. Score should be -inf or 0?
        # The contract says: "A centre on or outside the square ... make the candidate invalid."
        # So we must ensure coords stay in [-0.5, 0.5].
        # My moves will clip to bounds.
        
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        
        r = min(min_pair / 2.0, min_wall)
        return r

    def propose_move(coords, step_size):
        """
        Pick a random point and jitter it.
        """
        new_coords = coords.copy()
        idx = np.random.randint(N)
        # Random direction
        angle = np.random.uniform(0, 2 * np.pi)
        # Random magnitude, scaled by step_size
        mag = np.random.uniform(0, step_size)
        dx = mag * np.cos(angle)
        dy = mag * np.sin(angle)
        
        new_coords[idx, 0] += dx
        new_coords[idx, 1] += dy
        
        # Clip to bounds
        new_coords[:, 0] = np.clip(new_coords[:, 0], -0.5, 0.5)
        new_coords[:, 1] = np.clip(new_coords[:, 1], -0.5, 0.5)
        
        return new_coords

    # Initial configuration: Random
    # A grid might be a better start for SA to avoid initial overlaps?
    # Let's start with a grid to have a decent initial score.
    # 50 points. 7x7 grid = 49. Add 1 more? Or 5x10?
    # Let's just use random for simplicity, SA is robust.
    # Actually, starting with a grid is smarter.
    # Grid 7x7 covers [-0.5, 0.5] with spacing 1/6 = 0.166.
    # Radius would be ~ 0.083. This is better than random.
    
    # Let's create a grid of 7x7 = 49 points, and place the 50th in the center? No, center is occupied.
    # Let's just use random initialization, it's safer and simpler to code correctly.
    coords = np.random.uniform(-0.4, 0.4, (N, 2))
    current_score = get_score(coords)
    best_coords = coords.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 0.05
    T_end = 0.0001
    cooling = 0.99
    step_size = 0.05
    
    start_time = time.time()
    
    # We have ~120s.
    # Each step involves O(N^2) distance calc.
    # N=50 -> 2500 pairs.
    # 100k steps * 2500 * 2 (float ops) ~ 500M ops. NumPy can do ~100-500M simple ops/sec.
    # So 100k steps might take 1-5 seconds. We can afford 500k steps.
    
    T = T_start
    while T > T_end:
        if time.time() - start_time > 110:
            break
            
        new_coords = propose_move(coords, step_size)
        new_score = get_score(new_coords)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            coords = new_coords
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_coords = coords.copy()
        
        T *= cooling
        
        # Reduce step size slightly as T cools?
        # step_size *= 0.999? Or keep constant. Constant is fine.
        
    return best_coords
