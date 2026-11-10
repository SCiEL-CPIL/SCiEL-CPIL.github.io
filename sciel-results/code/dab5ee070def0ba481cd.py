import numpy as np
import itertools
import time
import random

def no_5_on_a_sphere8() -> np.ndarray:
        n = 8
        grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
        grid_sq = (grid ** 2).sum(axis=1)
        N = len(grid)
        
        def can_add(p_idx, S_indices, S_points):
            m = len(S_indices)
            if m < 4:
                return True
            
            p = grid[p_idx]
            p_sq = grid_sq[p_idx]
            
            W = np.empty((m, 4), dtype=np.float64)
            W[:, 0:3] = S_points - p
            W[:, 3] = (grid_sq[S_indices] - p_sq)
            
            if m < 20:
                # Small m, just loop to avoid overhead of creating large combo arrays
                # Actually, even for m=20, combos=4845.
                # Let's just use the batch method always, it's cleaner.
                pass
            
            combos = list(itertools.combinations(range(m), 4))
            if not combos:
                return True
            idx_array = np.array(combos, dtype=np.int32)
            sub = W[idx_array]
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)

        # Simulated Annealing
        start_time = time.time()
        time_limit = 550
        
        best_set = []
        best_size = 0
        
        # Initialize with a greedy set
        current_indices = []
        current_points = np.empty((0, 3), dtype=np.int64)
        in_set = set()
        
        # Greedy Init
        np.random.seed(42)
        order = np.random.permutation(N)
        for idx in order:
            if time.time() - start_time > time_limit: break
            if can_add(idx, current_indices, current_points):
                current_indices.append(idx)
                current_points = np.vstack([current_points, grid[idx]]) if len(current_points) else grid[idx][None, :]
                in_set.add(idx)
                
        best_size = len(current_indices)
        best_set = list(current_indices)
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                break
                
            # Pick a move
            # 1. Try to add a random point
            # 2. Try to remove a random point
            # 3. Try to swap
            
            move_type = np.random.rand()
            accepted = False
            
            if move_type < 0.5: # Add
                # Pick random point not in set
                # Efficient: pick random index, if in set, retry or pick from list of not-in-set
                # For simplicity, pick random index. If in set, skip.
                # Better: maintain a list of not-in-set? Or just try random.
                # N=512. Probability of hitting in-set is m/512.
                # If m=50, prob ~10%. Acceptable.
                
                p_idx = np.random.randint(N)
                if p_idx in in_set:
                    continue
                
                if can_add(p_idx, current_indices, current_points):
                    # Accept
                    current_indices.append(p_idx)
                    current_points = np.vstack([current_points, grid[p_idx]])
                    in_set.add(p_idx)
                    accepted = True
                    if len(current_indices) > best_size:
                        best_size = len(current_indices)
                        best_set = list(current_indices)
                        
            elif move_type < 0.8: # Remove
                if len(current_indices) == 0:
                    continue
                # Remove random point
                rem_idx_pos = np.random.randint(len(current_indices))
                rem_idx = current_indices[rem_idx_pos]
                
                # Delta E = -1 (size decreases)
                # Accept if random < exp(-1/T)
                if np.random.rand() < np.exp(-1.0 / T):
                    # Remove
                    current_indices.pop(rem_idx_pos)
                    current_points = np.delete(current_points, rem_idx_pos, axis=0)
                    in_set.remove(rem_idx)
                    accepted = True
                    
            else: # Swap
                if len(current_indices) == 0:
                    continue
                
                rem_idx_pos = np.random.randint(len(current_indices))
                rem_idx = current_indices[rem_idx_pos]
                
                add_idx = np.random.randint(N)
                if add_idx in in_set:
                    continue
                
                # Check if S \ {rem} + {add} is valid
                # This means: 
                # 1. add is valid with respect to S \ {rem}
                # 2. (implicitly, the rest of S is already valid)
                
                # We need to check can_add(add_idx, S \ {rem})
                # Create temporary lists?
                # Or modify can_add to take a mask?
                # For now, just create temp lists. m is small.
                
                temp_indices = [i for j, i in enumerate(current_indices) if j != rem_idx_pos]
                temp_points = np.delete(current_points, rem_idx_pos, axis=0)
                
                if can_add(add_idx, temp_indices, temp_points):
                    # Accept with prob 1? 
                    # Size is same. 
                    # We accept to explore.
                    # If we accept, we replace rem with add.
                    current_indices[rem_idx_pos] = add_idx
                    current_points[rem_idx_pos] = grid[add_idx]
                    in_set.remove(rem_idx)
                    in_set.add(add_idx)
                    accepted = True
            
            if accepted:
                T *= cooling
            else:
                # If rejected, maybe cool slower or do nothing?
                # Standard SA: T always cools.
                T *= cooling

        if not best_set:
            return np.empty((0, 3), dtype=np.int64)
        return grid[best_set].astype(np.int64)
