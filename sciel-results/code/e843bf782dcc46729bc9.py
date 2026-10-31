"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Strategy:
    # Use vectors with norm squared = 4 (Norm = 2).
    # The constraint is min pairwise dist >= max norm.
    # If all vectors have norm 2, we need dist^2 >= 4.
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # 8 - 2 u.v >= 4 => u.v <= 2.
    
    # Type A: Weight 1 vectors in {-2, 0, 2}^11.
    # Norm^2 = 4.
    # Count: 11 positions * 2 signs = 22.
    # Mutual compatibility:
    # Same pos, diff sign: u.v = -4 <= 2.
    # Diff pos: u.v = 0 <= 2.
    # All 22 are compatible.
    
    # Type B: Weight 2 vectors in {-1, 0, 1}^11.
    # Norm^2 = 2.
    # Wait, if we mix Norm 2 and Norm sqrt(2), max norm is 2.
    # Constraint: dist^2 >= 4.
    # For u (Norm 2) and v (Norm sqrt(2)):
    # dist^2 = 4 + 2 - 2 u.v = 6 - 2 u.v.
    # 6 - 2 u.v >= 4 => 2 u.v <= 2 => u.v <= 1.
    
    # Let's check Type B vs Type B (both Norm sqrt(2)):
    # dist^2 = 2 + 2 - 2 u.v = 4 - 2 u.v.
    # 4 - 2 u.v >= 4 => -2 u.v >= 0 => u.v <= 0.
    
    # So Type B vectors must have dot product <= 0 with each other.
    # And Type B vs Type A must have dot product <= 1.
    
    # Let's re-evaluate Type B mutual compatibility.
    # u = (1, 1, 0...), v = (1, -1, 0...). u.v = 0. OK.
    # u = (1, 1, 0...), v = (1, 0, 1...). u.v = 1. FAIL (1 > 0).
    # So Type B vectors are NOT mutually compatible if they share a coordinate with same sign.
    
    # Therefore, we cannot just take ALL Type B vectors.
    # We need an independent set for Type B with constraint u.v <= 0.
    # And we need to check Type B vs Type A.
    
    # Let's simplify.
    # Just take Type A (22 vectors).
    # And try to find a large set of Type B vectors compatible with each other and Type A.
    
    # Actually, let's look at Type C: Weight 4 in {-1, 0, 1}^11.
    # We established that W4 vs W4 requires u.v <= 2.
    # W4 vs W1 (Type A) requires u.v <= 2. (Always true for W4, max dot 2).
    # W4 vs W2 (Type B) requires u.v <= 1.
    
    # Given the complexity of mixing, let's stick to the safest high-yield set.
    # The inspiration 2 got ~193 W4 vectors.
    # If we take those 193 W4 vectors, we can add ALL 22 W1 vectors.
    # Total ~215.
    
    # Can we do better?
    # What if we take NO W4, but a large set of W2?
    # W2 vs W2 requires u.v <= 0.
    # This is equivalent to finding a set of edges in K11 such that no two edges share a vertex with the same "sign orientation"?
    # Actually, u.v <= 0 for W2 vectors means they cannot share a coordinate with the same sign.
    # This is like a matching in a signed graph.
    # Max number of W2 vectors with u.v <= 0?
    # We can take all vectors with support {i, j} and signs (+, -) or (-, +)?
    # u=(1, -1, 0..), v=(1, 0, -1..). u.v = 1. Fail.
    
    # Let's just use the W4 + W1 strategy, which is proven to work and yields ~215.
    # We will implement the fast MIS for W4.
    
    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix for W4
    # Conflict if u.v > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Compute degrees
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    # Simulated Annealing for Maximum Independent Set
    # with dynamic vertex weighting based on local density
    
    rng = np.random.default_rng(42)
    
    def greedy_mis(conflicts, deadline, rng, degrees):
        """
        Greedy Maximum Independent Set with degree-based ordering.
        """
        N = len(conflicts)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        selected_indices = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
                
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
                
            # Find min degree among alive
            min_deg = np.min(deg[alive_indices])
            candidates = alive_indices[deg[alive_indices] == min_deg]
            
            # Random tie-breaking
            chosen = candidates[rng.integers(len(candidates))]
            selected_indices.append(chosen)
            
            # Find neighbors to remove
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            
            # Update alive
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            remaining_alive = np.where(alive)[0]
            if len(remaining_alive) > 0:
                removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
                deg[remaining_alive] -= removed_conflicts

        return selected_indices
    
    def simulated_annealing_mis(conflicts, deadline, rng, degrees):
        """
        Simulated Annealing for Maximum Independent Set.
        Uses dynamic vertex weighting based on local density.
        """
        N = len(conflicts)
        
        # Initialize with greedy solution
        initial_set = greedy_mis(conflicts, min(deadline, time.time() + 5), rng, degrees)
        
        if not initial_set:
            return []
        
        current_selected = set(initial_set)
        current_size = len(current_selected)
        best_selected = set(initial_set)
        best_size = current_size
        
        # Precompute conflict sets for faster lookup
        conflict_sets = [np.where(conflicts[i])[0] for i in range(N)]
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        cooling_rate = 0.995
        
        T = T_start
        step = 0
        
        while time.time() < deadline and T > T_end:
            step += 1
            
            # Choose a random move: add, remove, or swap
            move_type = rng.integers(3)
            
            if move_type == 0:  # Add a vertex
                # Find vertices that can be added (no conflicts with current set)
                if current_selected:
                    # A vertex can be added if it has no conflicts with any selected vertex
                    # Compute conflicts with current set
                    can_add = np.ones(N, dtype=bool)
                    for v in current_selected:
                        can_add[conflict_sets[v]] = False
                        can_add[v] = False
                    candidates = np.where(can_add)[0]
                else:
                    candidates = np.arange(N)
                
                if len(candidates) > 0:
                    v_add = candidates[rng.integers(len(candidates))]
                    new_set = current_selected | {v_add}
                    new_size = len(new_set)
                    
                    # Acceptance: always accept if improves, else with probability exp(-delta/T)
                    delta = current_size - new_size  # negative if improves
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_selected = new_set
                        current_size = new_size
                        if current_size > best_size:
                            best_selected = set(current_selected)
                            best_size = current_size
                            
            elif move_type == 1:  # Remove a vertex
                if current_selected:
                    v_remove = list(current_selected)[rng.integers(len(current_selected))]
                    new_set = current_selected - {v_remove}
                    new_size = len(new_set)
                    
                    delta = current_size - new_size  # positive if worsens
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_selected = new_set
                        current_size = new_size
                        if current_size > best_size:
                            best_selected = set(current_selected)
                            best_size = current_size
                            
            else:  # Swap: remove one, add another
                if current_selected:
                    v_remove = list(current_selected)[rng.integers(len(current_selected))]
                    temp_set = current_selected - {v_remove}
                    
                    # Find vertices that can be added
                    if temp_set:
                        can_add = np.ones(N, dtype=bool)
                        for v in temp_set:
                            can_add[conflict_sets[v]] = False
                            can_add[v] = False
                        candidates = np.where(can_add)[0]
                    else:
                        candidates = np.arange(N)
                    
                    if len(candidates) > 0:
                        v_add = candidates[rng.integers(len(candidates))]
                        new_set = temp_set | {v_add}
                        new_size = len(new_set)
                        
                        delta = current_size - new_size
                        if delta < 0 or rng.random() < np.exp(-delta / T):
                            current_selected = new_set
                            current_size = new_size
                            if current_size > best_size:
                                best_selected = set(current_selected)
                                best_size = current_size
            
            # Cool down
            T *= cooling_rate
            
            # Every 100 steps, try to re-greedy from current state
            if step % 100 == 0 and time.time() < deadline - 5:
                # Convert current set to a mask and run greedy completion
                mask = np.zeros(N, dtype=bool)
                for v in current_selected:
                    mask[v] = True
                
                # Try to add more vertices greedily
                alive = ~mask
                while np.any(alive):
                    if time.time() > deadline - 5:
                        break
                    alive_indices = np.where(alive)[0]
                    if len(alive_indices) == 0:
                        break
                    
                    # Find min degree among alive (considering only conflicts with alive)
                    # Approximate: use original degrees
                    min_deg = np.min(degrees[alive_indices])
                    candidates = alive_indices[degrees[alive_indices] == min_deg]
                    chosen = candidates[rng.integers(len(candidates))]
                    
                    # Check if chosen conflicts with any in current set
                    conflicts_with_current = np.any([conflicts[chosen, v] for v in current_selected]) if current_selected else False
                    
                    if not conflicts_with_current:
                        current_selected.add(chosen)
                        current_size += 1
                        mask[chosen] = True
                        alive[chosen] = False
                        # Remove neighbors from alive
                        neighbors = np.where(conflicts[chosen])[0]
                        alive[neighbors] = False
                        
                        if current_size > best_size:
                            best_selected = set(current_selected)
                            best_size = current_size
                    else:
                        alive[chosen] = False
        
        return list(best_selected)
    
    # Run simulated annealing
    sa_deadline = deadline - 10
    sa_result = simulated_annealing_mis(conflicts, sa_deadline, rng, degrees)
    
    # Also run a few greedy restarts to compare
    best_greedy = []
    for _ in range(3):
        if time.time() > sa_deadline - 2:
            break
        greedy_result = greedy_mis(conflicts, min(sa_deadline - 1, time.time() + 3), rng, degrees)
        if len(greedy_result) > len(best_greedy):
            best_greedy = greedy_result
    
    # Use the better of SA and greedy
    if len(sa_result) > len(best_greedy):
        selected_indices = sa_result
    else:
        selected_indices = best_greedy
    
    if not selected_indices:
        return vectors_v1
    
    result_v4 = vectors_v4[selected_indices]
    
    # Combine with W1
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
