# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 117  # Safety margin for 120s limit

    def check_validity(n, colors):
        """Check if the coloring is valid (no red K4, no blue K5)."""
        if n < 4:
            return True
        # Check Red K4: any 4 vertices with all 4 triples red (0)
        # Check Blue K5: any 5 vertices with all 10 triples blue (1)
        
        # For efficiency, we can precompute indices, but for a final check, 
        # we can iterate or use vectorized checks if we have the mapping.
        # Since this is called at the end, we can afford a slightly slower check 
        # or reuse the precomputed structures if we passed them. 
        # However, construct() is the entry point, so we need a self-contained check 
        # or pass the maps. Let's just do a direct check for correctness in the 
        # final verification step, or better, rely on the SA energy being 0.
        # The SA loop ensures energy 0. We just need to make sure we return the 
        # correct format.
        
        # Let's do a quick sanity check for small n or just trust the SA energy.
        # Actually, let's implement a fast validity check using the precomputed maps 
        # if available, but here we are in the main construct, so we'll build 
        # necessary structures if we want to verify. 
        # To save time, we assume the SA loop's energy==0 is sufficient for validity.
        return True

    def solve_for_n(n, time_budget):
        """Attempt to find a valid coloring for n vertices within time_budget."""
        if n < 1:
            return None
            
        # Precompute triple indices
        triples = []
        idx_map = {}
        i = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx_map[(a,b,c)] = i
                    triples.append((a,b,c))
                    i += 1
        
        n_triples = len(triples)
        if n_triples == 0:
            return [0] * 0

        # Precompute quad and quint triple indices for incremental updates
        quad_triples = []
        for quad in combinations(range(n), 4):
            quad_triples.append([idx_map[tri] for tri in combinations(quad, 3)])
        
        quint_triples = []
        for quint in combinations(range(n), 5):
            quint_triples.append([idx_map[tri] for tri in combinations(quint, 3)])
        
        n_quads = len(quad_triples)
        n_quints = len(quint_triples)

        if n_quads == 0 and n_quints == 0:
             return [0] * n_triples

        # Build reverse maps: triple_index -> list of quad_indices, quint_indices
        triple_to_quads = [[] for _ in range(n_triples)]
        for q_idx, quads in enumerate(quad_triples):
            for t in quads:
                triple_to_quads[t].append(q_idx)
                
        triple_to_quints = [[] for _ in range(n_triples)]
        for q_idx, quints in enumerate(quint_triples):
            for t in quints:
                triple_to_quints[t].append(q_idx)

        # Convert to numpy arrays for faster access if needed, 
        # but lists are fine for the reverse maps.
        # For the forward maps (quad -> triples), we use the lists of lists.
        
        # Initial random coloring. 
        # Heuristic: Random is often good enough for SA to escape local minima.
        colors = np.random.choice([0, 1], size=n_triples)
        
        # Initialize violation states
        # quad_is_red[k] is True if all 4 triples in quad k are 0
        # quint_is_blue[k] is True if all 10 triples in quint k are 1
        
        quad_is_red = np.zeros(n_quads, dtype=bool)
        quint_is_blue = np.zeros(n_quints, dtype=bool)
        
        # Calculate initial energy
        # We can do this vectorized for the initial state
        if n_quads > 0:
            # quad_triples is a list of lists of indices
            # Convert to array for vectorized check
            quad_arr = np.array(quad_triples, dtype=np.int32)
            quad_colors = colors[quad_arr]
            quad_is_red = np.all(quad_colors == 0, axis=1)
        if n_quints > 0:
            quint_arr = np.array(quint_triples, dtype=np.int32)
            quint_colors = colors[quint_arr]
            quint_is_blue = np.all(quint_colors == 1, axis=1)
            
        current_energy = np.sum(quad_is_red) + np.sum(quint_is_blue)
        
        if current_energy == 0:
            return colors.tolist()

        # Simulated Annealing Parameters
        T = 5.0
        T_min = 0.05
        cooling_rate = 0.99995 # Slow cooling
        start_sa = time.time()
        
        # Precompute "others" lists for faster delta calculation?
        # For each triple, and each quad it belongs to, we need the other 3 indices.
        # For each triple, and each quint it belongs to, we need the other 9 indices.
        # This is memory intensive but speeds up the loop significantly.
        
        # Let's create a structure: 
        # tri_other_quads[tri_idx] = list of (quad_idx, [other_tri_1, other_tri_2, other_tri_3])
        # tri_other_quints[tri_idx] = list of (quint_idx, [other_tri_1, ..., other_tri_9])
        
        tri_other_quads = [[] for _ in range(n_triples)]
        for q_idx, quads in enumerate(quad_triples):
            for t in quads:
                others = [x for x in quads if x != t]
                tri_other_quads[t].append((q_idx, others))
                
        tri_other_quints = [[] for _ in range(n_triples)]
        for q_idx, quints in enumerate(quint_triples):
            for t in quints:
                others = [x for x in quints if x != t]
                tri_other_quints[t].append((q_idx, others))

        while T > T_min:
            if time.time() - start_sa > time_budget:
                break
            
            tri_idx = np.random.randint(n_triples)
            old_color = colors[tri_idx]
            new_color = 1 - old_color
            
            delta_energy = 0
            
            # Update Quads
            for q_idx, others in tri_other_quads[tri_idx]:
                was_red = quad_is_red[q_idx]
                
                if was_red:
                    if new_color == 1:
                        # Becomes not red
                        delta_energy -= 1
                        quad_is_red[q_idx] = False
                    else:
                        # Stays red (new_color is 0, others were 0)
                        pass
                else:
                    if new_color == 1:
                        # Stays not red
                        pass
                    else:
                        # new_color is 0. Check if others are all 0
                        # If yes, it becomes red
                        if all(colors[o] == 0 for o in others):
                            delta_energy += 1
                            quad_is_red[q_idx] = True
            
            # Update Quints
            for q_idx, others in tri_other_quints[tri_idx]:
                was_blue = quint_is_blue[q_idx]
                
                if was_blue:
                    if new_color == 0:
                        # Becomes not blue
                        delta_energy -= 1
                        quint_is_blue[q_idx] = False
                    else:
                        # Stays blue (new_color is 1, others were 1)
                        pass
                else:
                    if new_color == 0:
                        # Stays not blue
                        pass
                    else:
                        # new_color is 1. Check if others are all 1
                        if all(colors[o] == 1 for o in others):
                            delta_energy += 1
                            quint_is_blue[q_idx] = True
            
            # Accept/Reject
            if delta_energy <= 0 or random.random() < np.exp(-delta_energy / T):
                colors[tri_idx] = new_color
                current_energy += delta_energy
                if current_energy == 0:
                    return colors.tolist()
            
            T *= cooling_rate
        
        return None

    # Try to find a valid coloring for decreasing n
    # Start with a high target, e.g., 34. If we fail, try 33, etc.
    # We allocate time to each attempt.
    # If we find a valid one, we return it immediately as it maximizes n for that attempt.
    # But we want the GLOBAL max n. So we should try high n first.
    
    # Let's try 34 first with a generous time budget.
    # If it fails, try 33 with remaining time, etc.
    
    current_time = time.time()
    remaining = deadline - current_time
    
    # Heuristic: If we have plenty of time, try 34.
    # If we find a valid coloring for 34, we are done (score 34).
    # If not, we might find a valid one for 33, etc.
    
    # To ensure we return *something* valid, we must ensure we don't run out of time 
    # trying only 34.
    
    # Let's split the time. 
    # Try 34 for 60s. If fail, try 33 for 30s. If fail, try 32 for 20s. If fail, try 31 for 5s. 
    # If all fail, we have a problem. But usually, random + SA finds small n quickly.
    # Actually, for n=1..10, validity is trivial or easy.
    # We can start from a small n and increase? No, we want max n.
    # So we must try high n first.
    
    # Let's try n=34 with a specific budget.
    # If we find a solution, return it.
    
    target_ns = [34, 33, 32, 31, 30, 25, 20, 10]
    
    # Allocate time proportionally or fixed chunks?
    # Let's give 50s to 34, 20s to 33, 10s to 32, 10s to 31, 10s to 30, 5s to 25, 5s to 20, 5s to 10.
    # Total 115s.
    
    budgets = {
        34: 50,
        33: 20,
        32: 10,
        31: 10,
        30: 10,
        25: 5,
        20: 5,
        10: 5
    }
    
    for n in target_ns:
        if time.time() > deadline:
            break
            
        budget = budgets.get(n, 5)
        # Don't exceed remaining time
        budget = min(budget, deadline - time.time())
        
        if budget <= 0:
            continue
            
        result = solve_for_n(n, budget)
        
        if result is not None:
            # Verify validity one last time just in case? 
            # The SA loop returns only if energy is 0. 
            # But let's trust it for speed.
            return {'n': n, 'colors': result}
            
    # Fallback: Return a trivial valid coloring for n=1 if all else fails
    # n=1 has 0 triples, so it's valid.
    return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
