import random
import numpy as np

N = 64

def construct():
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    coords = np.array(points)
    
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sum(diff**2, axis=2).astype(np.int32)
    
    def can_add(p_idx, S_indices, dist_sets):
        if not S_indices:
            return True
        d_p_S = D[p_idx, S_indices]
        if len(np.unique(d_p_S)) < len(d_p_S):
            return False
        for i, a_idx in enumerate(S_indices):
            if d_p_S[i] in dist_sets[a_idx]:
                return False
        return True

    def add_point(p_idx, S_indices, dist_sets):
        S_indices.append(p_idx)
        dist_sets[p_idx] = set()
        for a_idx in S_indices[:-1]:
            d = D[p_idx, a_idx]
            dist_sets[a_idx].add(d)
            dist_sets[p_idx].add(d)

    def remove_point(r_idx, S_indices, dist_sets):
        try:
            pos = S_indices.index(r_idx)
        except ValueError:
            return
        S_indices.pop(pos)
        for a_idx in S_indices:
            d = D[r_idx, a_idx]
            dist_sets[a_idx].discard(d)
        del dist_sets[r_idx]

    def optimize(S_indices, dist_sets):
        # Greedily add points
        # We need an order. Random is okay.
        # To make it deterministic-ish for reproducibility in restarts, we can use a global random state.
        order = list(range(num_points))
        random.shuffle(order)
        for p_idx in order:
            if p_idx not in dist_sets:
                if can_add(p_idx, S_indices, dist_sets):
                    add_point(p_idx, S_indices, dist_sets)
        return S_indices, dist_sets

    best_size = 0
    best_S = []
    
    # ILS
    for restart in range(200):
        random.seed(restart)
        # Initialize with a small random set or empty
        S_indices = []
        dist_sets = {}
        
        # Add a few random points to seed
        init_points = random.sample(range(num_points), 5)
        for p in init_points:
            if can_add(p, S_indices, dist_sets):
                add_point(p, S_indices, dist_sets)
        
        S_indices, dist_sets = optimize(S_indices, dist_sets)
        current_size = len(S_indices)
        
        if current_size > best_size:
            best_size = current_size
            best_S = S_indices[:]
            
        # Perturb and re-optimize
        for step in range(100):
            # Remove 1-3 random points
            k = random.randint(1, 3)
            for _ in range(k):
                if S_indices:
                    r = random.choice(S_indices)
                    remove_point(r, S_indices, dist_sets)
            
            S_indices, dist_sets = optimize(S_indices, dist_sets)
            current_size = len(S_indices)
            
            if current_size > best_size:
                best_size = current_size
                best_S = S_indices[:]
                
    best_set = [points[idx] for idx in best_S]
    return best_set
