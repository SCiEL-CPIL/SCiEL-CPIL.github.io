"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a 4-coloring of K_n with no monochromatic C6, maximizing n.
    
    Strategy:
    1. Start with known good structures or random initializations.
    2. Use simulated annealing / local search to minimize C6 violations.
    3. Try to grow n from smaller valid colorings.
    
    Time limit: ~140 seconds (safety margin before 150s kill).
    """
    start_time = time.time()
    TIME_LIMIT = 140.0
    
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)
    
    # Precompute all C6 cycles for given n
    # A C6 is a set of 6 vertices in a cycle: v0-v1-v2-v3-v4-v5-v0
    # We need to check if all 6 edges have the same color.
    
    def generate_c6_cycles(n):
        """Generate all distinct 6-cycles in K_n.
        A 6-cycle is determined by 6 vertices in order up to rotation and reversal.
        Number of 6-cycles in K_n = C(n,6) * (6-1)!/2 = C(n,6) * 60
        For n=17: C(17,6)=12376, *60 = 742560 cycles. That's a lot but manageable.
        For n=18: C(18,6)=18564, *60 = 1113840
        For n=19: C(19,6)=27132, *60 = 1627920
        """
        cycles = []
        vertices = list(range(n))
        # To reduce duplicates, fix v0=0, v1=1, and enumerate permutations of remaining 4
        # Actually, let's just enumerate all 6-subsets and all cyclic orderings
        # More efficient: fix first two vertices to reduce symmetry
        from itertools import combinations, permutations
        
        # For each 6-subset, there are (6-1)!/2 = 60 distinct cycles
        # But we can fix v0 and v1 to reduce: fix v0=min, v1=second_min, then permute rest
        # Actually simpler: just enumerate all 6-subsets and all 30 distinct cycle orderings
        # (fix first vertex, then 5! / 2 = 60... no, (6-1)!/2 = 60)
        
        # Let's just do it directly for clarity, but be smart about it
        # For n up to 19, C(n,6) is at most 27132, and 60 cycles each = 1.6M total
        # That's fine.
        
        for subset in combinations(range(n), 6):
            # Fix v0 = subset[0], v1 = subset[1] to break rotational/reflection symmetry
            # Then permute the remaining 4 vertices in positions 2,3,4,5
            # But we also need to account for the fact that reversing the cycle gives the same cycle
            # With v0 and v1 fixed, the cycle is v0-v1-v2-v3-v4-v5-v0
            # Reversal would be v0-v5-v4-v3-v2-v1-v0, which has v0 fixed but v1 becomes v5
            # So with v0 and v1 fixed, each distinct cycle is counted exactly once
            # Wait, that's not right. Let me think again.
            # A cycle on 6 vertices has 12 symmetries (6 rotations * 2 reflections).
            # By fixing v0 and v1, we break all 12 symmetries? 
            # Rotations: v0 can be any of 6 positions, v1 can be adjacent (2 choices) -> 12
            # So fixing v0 and v1 to specific positions in the cycle breaks all symmetries.
            # So for each 6-subset, the number of distinct cycles with v0 and v1 fixed is 4! = 24
            # But wait, we're fixing which vertices are v0 and v1 (the smallest two in the subset)
            # Then we permute the other 4 in positions 2-5: 4! = 24 permutations
            # Each gives a distinct cycle? Let's verify: 
            # Cycle: v0-v1-a-b-c-d-v0. If we swap a and b, we get v0-v1-b-a-c-d-v0, different.
            # Yes, 24 distinct cycles per 6-subset.
            
            rest = list(subset[2:])
            for perm in permutations(rest):
                cycle = (subset[0], subset[1], perm[0], perm[1], perm[2], perm[3])
                # Store the 6 edges of this cycle
                edges = []
                for k in range(6):
                    u = cycle[k]
                    v = cycle[(k+1) % 6]
                    if u > v:
                        u, v = v, u
                    edges.append((u, v))
                cycles.append(edges)
        
        return cycles
    
    # Precompute edge index mapping
    def build_edge_index(n):
        """Map (i,j) with i<j to index in the color array."""
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_idx[(i,j)] = idx
                idx += 1
        return edge_idx
    
    def count_c6_violations(colors, n, cycles, edge_idx):
        """Count number of monochromatic C6 cycles."""
        violations = 0
        for edges in cycles:
            # Get colors of the 6 edges
            c0 = colors[edge_idx[edges[0]]]
            c1 = colors[edge_idx[edges[1]]]
            c2 = colors[edge_idx[edges[2]]]
            c3 = colors[edge_idx[edges[3]]]
            c4 = colors[edge_idx[edges[4]]]
            c5 = colors[edge_idx[edges[5]]]
            if c0 == c1 == c2 == c3 == c4 == c5:
                violations += 1
        return violations
    
    def random_coloring(n):
        m = n * (n - 1) // 2
        return np.random.randint(0, 4, size=m)
    
    def try_local_search(n, max_iters=None, seed=None):
        """Local search to eliminate all C6 violations."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        edge_idx = build_edge_index(n)
        cycles = generate_c6_cycles(n)
        m = n * (n - 1) // 2
        
        # Initialize
        colors = random_coloring(n)
        violations = count_c6_violations(colors, n, cycles, edge_idx)
        
        if violations == 0:
            return colors, True
        
        best_colors = colors.copy()
        best_violations = violations
        
        # Simulated annealing
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        iters = 0
        while time_remaining() > 10.0:
            # Pick a random edge to flip
            eidx = rng.randint(0, m - 1)
            old_color = colors[eidx]
            new_color = (old_color + rng.randint(1, 3)) % 4
            
            # Compute delta in violations
            # Only cycles containing this edge can change
            # For efficiency, we could precompute which cycles contain which edge
            # But for now, let's just recompute (slow but simple)
            # Actually, let's be smarter: track violations incrementally
            
            colors[eidx] = new_color
            new_violations = count_c6_violations(colors, n, cycles, edge_idx)
            delta = new_violations - violations
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                violations = new_violations
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors.copy()
                    if violations == 0:
                        return best_colors, True
            else:
                colors[eidx] = old_color
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            iters += 1
            if max_iters and iters >= max_iters:
                break
        
        return best_colors, best_violations == 0
    
    # Precompute cycles for various n values
    # Start with n=17 (known to be achievable)
    
    best_result = {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
    
    # Try to find valid colorings for increasing n
    for target_n in [17, 18, 19]:
        if time_remaining() < 30.0:
            break
        
        print(f"Trying n={target_n}...")
        
        # Precompute structures
        edge_idx = build_edge_index(target_n)
        cycles = generate_c6_cycles(target_n)
        m = target_n * (target_n - 1) // 2
        
        # Try multiple random restarts
        for attempt in range(50):
            if time_remaining() < 15.0:
                break
            
            rng_seed = attempt * 12345 + 777
            rng = random.Random(rng_seed)
            np.random.seed(rng_seed)
            
            colors = random_coloring(target_n)
            violations = count_c6_violations(colors, target_n, cycles, edge_idx)
            
            if violations == 0:
                best_result = {'n': target_n, 'colors': colors.tolist()}
                break
            
            # Local search with simulated annealing
            T = 1.0
            cooling = 0.9998
            min_T = 0.005
            best_colors = colors.copy()
            best_violations = violations
            
            stuck_counter = 0
            while time_remaining() > 15.0 and best_violations > 0:
                eidx = rng.randint(0, m - 1)
                old_color = colors[eidx]
                new_color = (old_color + rng.randint(1, 3)) % 4
                
                colors[eidx] = new_color
                new_violations = count_c6_violations(colors, target_n, cycles, edge_idx)
                delta = new_violations - violations
                
                if delta < 0 or rng.random() < np.exp(-max(0, delta) / T):
                    violations = new_violations
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors.copy()
                        stuck_counter = 0
                        if violations == 0:
                            break
                else:
                    colors[eidx] = old_color
                
                T *= cooling
                if T < min_T:
                    T = min_T
                    # Restart with higher temperature occasionally
                    if stuck_counter > 5000:
                        T = 0.5
                        stuck_counter = 0
                        # Random restart
                        colors = random_coloring(target_n)
                        violations = count_c6_violations(colors, target_n, cycles, edge_idx)
                        best_violations = min(best_violations, violations)
                        if violations < best_violations:
                            best_violations = violations
                            best_colors = colors.copy()
                else:
                    stuck_counter += 1
            
            if best_violations == 0:
                best_result = {'n': target_n, 'colors': best_colors.tolist()}
                break
        
        if best_result['n'] >= target_n:
            break
    
    # Verify the result
    if best_result['n'] > 1:
        n = best_result['n']
        colors = np.array(best_result['colors'])
        edge_idx = build_edge_index(n)
        cycles = generate_c6_cycles(n)
        violations = count_c6_violations(colors, n, cycles, edge_idx)
        if violations > 0:
            # Fall back to the known valid n=9 coloring
            n = 9
            colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
            best_result = {'n': n, 'colors': colors}
    
    return best_result

# EVOLVE-BLOCK-END
