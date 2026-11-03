import time
import math
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Constants
    # Inradius of unit hexagon
    r_in = math.sqrt(3) / 2.0
    # Minimum distance between centers for disjoint unit hexagons (aligned)
    d_min = 2.0 * r_in  # sqrt(3)
    
    def calc_required_S(x, y):
        """
        Calculate the minimum side length S of a regular hexagon 
        (centered at 0,0, flat-top) that contains point (x,y).
        """
        ax = abs(x)
        ay = abs(y)
        # Condition 1: |x| <= sqrt(3)/2 * S  => S >= 2/sqrt(3) * |x|
        S1 = (2.0 / math.sqrt(3.0)) * ax
        # Condition 2: |y| <= S - |x|/sqrt(3) => S >= |y| + |x|/sqrt(3)
        S2 = ay + ax / math.sqrt(3.0)
        return max(S1, S2)

    def check_validity(positions, S):
        """
        Check if all positions are inside hexagon of side S 
        and if any two are closer than d_min.
        Returns (is_valid, max_overlap).
        """
        max_overlap = 0.0
        for i in range(len(positions)):
            x, y = positions[i]
            req_S = calc_required_S(x, y)
            if req_S > S:
                # If a point is outside, it's invalid for this S
                return False, 0.0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < d_min:
                    overlap = d_min - dist
                    if overlap > max_overlap:
                        max_overlap = overlap
        return max_overlap < 1e-6, max_overlap

    def local_search_for_S(S, initial_pos, max_iters=5000):
        """
        Try to find a valid packing for a given S starting from initial_pos.
        Uses simulated annealing / hill climbing to minimize overlap and out-of-bounds penalty.
        """
        pos = [list(p) for p in initial_pos]
        current_energy = 0.0
        
        def calc_energy():
            e = 0.0
            # Overlap penalty
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < d_min:
                        e += (d_min - dist) ** 2
            # Out of bounds penalty
            for i in range(len(pos)):
                req_S = calc_required_S(pos[i][0], pos[i][1])
                if req_S > S:
                    e += (req_S - S) ** 2 * 10.0 # Heavy penalty for being outside
            return e

        current_energy = calc_energy()
        best_pos = [list(p) for p in pos]
        best_energy = current_energy
        
        T = 1.0
        cooling = 0.995
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            i = random.randint(0, N - 1)
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            
            old_x, old_y = pos[i]
            pos[i][0] += dx
            pos[i][1] += dy
            
            new_energy = calc_energy()
            
            if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = [list(p) for p in pos]
            else:
                pos[i][0] = old_x
                pos[i][1] = old_y
                
            T *= cooling
            if T < 1e-3:
                break
        
        # Final check on best found
        is_valid, max_ov = check_validity(best_pos, S)
        if is_valid:
            return best_pos
        return None

    # Initial guesses
    # 1. Hexagonal grid
    def get_hex_grid_positions():
        positions = []
        # Center
        positions.append([0.0, 0.0])
        # Ring 1 (6)
        for k in range(6):
            angle = math.pi / 3 * k
            x = d_min * math.cos(angle)
            y = d_min * math.sin(angle)
            positions.append([x, y])
        # Ring 2 (we need 11, so we have 7, need 4 more)
        # Add 4 from Ring 2
        for k in range(4):
            angle = math.pi / 3 * k
            x = 2 * d_min * math.cos(angle)
            y = 2 * d_min * math.sin(angle)
            positions.append([x, y])
        return positions

    initial_guesses = [get_hex_grid_positions()]
    
    # Add a few random perturbations of the grid
    for _ in range(5):
        base = get_hex_grid_positions()
        perturbed = []
        for p in base:
            perturbed.append([p[0] + random.gauss(0, 0.2), p[1] + random.gauss(0, 0.2)])
        initial_guesses.append(perturbed)

    best_S = 20.0
    best_config = None

    # Binary search for minimal S
    low_S = 5.0
    high_S = 20.0
    
    # First, try to find a valid S with the grid
    # We'll iterate S downwards
    current_S = 15.0
    
    # Strategy: 
    # 1. Find a valid S using local search.
    # 2. Try to reduce S.
    
    found_valid_S = None
    # Try a range of S values
    S_candidates = [10.0, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0]
    
    for S_try in S_candidates:
        if time.time() > deadline - 10:
            break
        for init in initial_guesses:
            result = local_search_for_S(S_try, init, max_iters=2000)
            if result:
                found_valid_S = S_try
                best_config = result
                break
        if found_valid_S:
            break
            
    if not found_valid_S:
        # If no valid S found in candidates, try a larger S
        for S_try in [12.0, 14.0, 16.0]:
            if time.time() > deadline - 5:
                break
            for init in initial_guesses:
                result = local_search_for_S(S_try, init, max_iters=2000)
                if result:
                    found_valid_S = S_try
                    best_config = result
                    break
            if found_valid_S:
                break

    if not best_config:
        # Fallback: return a very large valid configuration
        # Just a line of hexagons
        positions = [[i * d_min, 0.0] for i in range(N)]
        max_req_S = max(calc_required_S(x, y) for x, y in positions)
        # Add margin
        return positions, (0.0, 0.0), max_req_S + 1.0, 0.0

    # Now we have a valid configuration for found_valid_S.
    # Try to polish/reduce S further.
    current_S = found_valid_S
    current_pos = best_config
    
    # Polish loop
    while time.time() < deadline - 5:
        improved = False
        # Try to reduce S slightly
        new_S = current_S * 0.98
        # Run local search to see if we can fit in new_S
        result = local_search_for_S(new_S, current_pos, max_iters=1000)
        if result:
            current_S = new_S
            current_pos = result
            improved = True
        else:
            # If we can't reduce, try a small random restart at current S to see if we can find a better config
            # that allows further reduction later? 
            # Or just break if we are stuck.
            # Let's try a few random restarts at current_S to see if we can find a "tighter" config
            # that might allow reduction.
            # Actually, if local search failed at new_S, it likely means the geometry is too tight.
            # We can try to optimize the positions for current_S to minimize "slack".
            # But for now, break to save time.
            break

    return current_pos, (0.0, 0.0), current_S, 0.0

# EVOLVE-BLOCK-END
