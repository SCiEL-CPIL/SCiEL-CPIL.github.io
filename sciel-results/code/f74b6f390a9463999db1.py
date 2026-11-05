"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

N = 16
TIME_LIMIT = 320  # seconds, safety margin below 330s

def construct():
    start_time = time.time()
    
    # Precompute all squared distances between grid points
    # There are N*N = 256 points
    points = list(itertools.product(range(N), repeat=2))
    n_pts = len(points)
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Precompute squared distance matrix
    dist2 = np.zeros((n_pts, n_pts), dtype=np.int32)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d = dx * dx + dy * dy
            dist2[i, j] = d
            dist2[j, i] = d
    
    # For each pair (i, j), precompute which points k would form an isosceles triangle
    # Triangle (i, j, k) is isosceles if any two of the three distances are equal
    # We'll build for each point i, a set of "conflict" relationships
    
    # Strategy: 
    # 1. Start with a greedy maximal set
    # 2. Try to remove points while maintaining maximality
    # 3. Use local search / simulated annealing to improve
    
    def is_isosceles(i, j, k):
        d1 = dist2[i, j]
        d2 = dist2[i, k]
        d3 = dist2[j, k]
        return d1 == d2 or d1 == d3 or d2 == d3
    
    def is_valid_set(indices):
        """Check if set of indices forms a no-isosceles set."""
        idx_list = list(indices)
        n = len(idx_list)
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    if is_isosceles(idx_list[a], idx_list[b], idx_list[c]):
                        return False
        return True
    
    def is_maximal(indices, all_points_set):
        """Check if no other point can be added."""
        idx_set = set(indices)
        for p in all_points_set:
            if p in idx_set:
                continue
            # Check if adding p creates any isosceles triangle
            for a, b in itertools.combinations(idx_set, 2):
                if is_isosceles(p, a, b):
                    break
            else:
                return False
        return True
    
    def greedy_maximal(order):
        """Build a maximal no-isosceles set greedily following the given order."""
        chosen = []
        chosen_set = set()
        for p in order:
            # Check if p can be added
            can_add = True
            for a, b in itertools.combinations(chosen, 2):
                if is_isosceles(p, a, b):
                    can_add = False
                    break
            if can_add:
                chosen.append(p)
                chosen_set.add(p)
        return chosen
    
    # Initial greedy construction in row-major order
    initial_order = list(range(n_pts))
    best_set = greedy_maximal(initial_order)
    best_size = len(best_set)
    
    # Try multiple random orders to find better initial solutions
    rng = random.Random(42)
    for _ in range(20):
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        order = list(range(n_pts))
        rng.shuffle(order)
        candidate = greedy_maximal(order)
        if len(candidate) < best_size:
            best_set = candidate
            best_size = len(candidate)
    
    # Local search: try to remove points while maintaining maximality
    def try_remove(set_indices):
        """Try to remove one point from the set while maintaining maximality."""
        idx_set = set(set_indices)
        for i in idx_set:
            if time.time() - start_time > TIME_LIMIT - 30:
                return None
            new_set = list(idx_set - {i})
            # Check if new_set is still valid (no isosceles) - it should be since we're removing
            # Check if new_set is still maximal
            if is_maximal(new_set, set(range(n_pts))):
                return new_set
        return None
    
    # Iterative improvement: try removing points
    current_set = list(best_set)
    improved = True
    while improved and time.time() - start_time < TIME_LIMIT - 60:
        improved = False
        for i in range(len(current_set)):
            if time.time() - start_time > TIME_LIMIT - 60:
                break
            idx_set = set(current_set)
            candidate = list(idx_set - {current_set[i]})
            if is_maximal(candidate, set(range(n_pts))):
                current_set = candidate
                best_set = candidate
                best_size = len(candidate)
                improved = True
                break
    
    # Simulated annealing: swap points to potentially find smaller maximal sets
    if time.time() - start_time < TIME_LIMIT - 90:
        current_set = list(best_set)
        current_size = len(current_set)
        temp = 1.0
        cooling = 0.95
        
        for iteration in range(200):
            if time.time() - start_time > TIME_LIMIT - 90:
                break
            
            # Pick a random point in current set to remove
            idx_in_set = rng.randint(0, len(current_set) - 1)
            removed_point = current_set[idx_in_set]
            
            # Find a random point not in current set to add
            not_in_set = [p for p in range(n_pts) if p not in set(current_set)]
            if not not_in_set:
                break
            added_point = rng.choice(not_in_set)
            
            new_set = current_set[:idx_in_set] + current_set[idx_in_set + 1:] + [added_point]
            
            # Check validity
            if not is_valid_set(new_set):
                continue
            
            # Check maximality
            if not is_maximal(new_set, set(range(n_pts))):
                continue
            
            new_size = len(new_set)
            delta = new_size - current_size
            
            if delta <= 0 or rng.random() < pow(temp, 1.0 / max(1, abs(delta))):
                current_set = new_set
                current_size = new_size
                if current_size < best_size:
                    best_set = current_set
                    best_size = current_size
            
            temp *= cooling
    
    # Final verification
    final_set = best_set
    if not is_valid_set(final_set):
        # Fallback to initial greedy
        final_set = greedy_maximal(list(range(n_pts)))
    
    # Convert indices back to points
    return [points[i] for i in final_set]


if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
