import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    def get_violation_counts(colors, n, triple_to_4, triple_to_5, four_sets, five_sets):
        """
        Compute the number of red K4s and blue K5s.
        colors: 0=Red, 1=Blue.
        Returns (num_red_k4, num_blue_k5)
        """
        # Vectorized counting is tricky with sparse indices, but for n<=34,
        # the number of sets is manageable.
        # Red K4: all 4 triples in a 4-set are Red (0).
        # Blue K5: all 10 triples in a 5-set are Blue (1).
        
        red_k4_count = 0
        blue_k5_count = 0
        
        # For efficiency in the main loop, we might precompute adjacency or use
        # incremental updates. But for the initial check and occasional full checks,
        # this is fine. However, in SA, we need fast cost evaluation.
        # We will maintain counts incrementally in the main loop.
        pass

    def solve_sa(n, time_limit):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        num_4sets = len(four_sets)
        num_5sets = len(five_sets)
        
        # Precompute membership:
        # triple_to_4sets[t_idx] = list of 4-set indices containing this triple
        # triple_to_5sets[t_idx] = list of 5-set indices containing this triple
        triple_to_4sets = [[] for _ in range(num_triples)]
        triple_to_5sets = [[] for _ in range(num_triples)]
        
        for i, s4 in enumerate(four_sets):
            for t in combinations(s4, 3):
                triple_to_4sets[triple_index[t]].append(i)
                
        for i, s5 in enumerate(five_sets):
            for t in combinations(s5, 3):
                triple_to_5sets[triple_index[t]].append(i)
                
        # We need to know, for each 4-set, which triples it contains, to update counts?
        # Actually, when we flip a triple, we know which 4-sets and 5-sets it affects.
        # We maintain:
        # red_count_4[i]: number of red triples in 4-set i
        # blue_count_5[j]: number of blue triples in 5-set j
        
        # Initialization: Random
        colors = np.random.randint(0, 2, size=num_triples)
        
        red_count_4 = np.zeros(num_4sets, dtype=np.int32)
        blue_count_5 = np.zeros(num_5sets, dtype=np.int32)
        
        # Initialize counts
        for t_idx in range(num_triples):
            if colors[t_idx] == 0: # Red
                for s4_idx in triple_to_4sets[t_idx]:
                    red_count_4[s4_idx] += 1
            else: # Blue
                for s5_idx in triple_to_5sets[t_idx]:
                    blue_count_5[s5_idx] += 1
                    
        # Calculate initial violations
        # Red violation: red_count_4[i] == 4
        # Blue violation: blue_count_5[j] == 10
        red_violations = np.sum(red_count_4 == 4)
        blue_violations = np.sum(blue_count_5 == 10)
        total_violations = red_violations + blue_violations
        
        if total_violations == 0:
            return colors.tolist()
            
        best_colors = colors.copy()
        best_violations = total_violations
        best_time = time.time()
        
        # Simulated Annealing Parameters
        T_start = 1.0
        T_end = 0.01
        alpha = 0.999  # Cooling rate
        
        # We want to maximize n. If we can't get 0 violations, we return None.
        # But we should keep the best valid coloring found.
        
        T = T_start
        no_improve_time = 0.0
        last_improve_time = time.time()
        
        # Energy function:
        # Primary: Violations (Red K4 + Blue K5). Minimize.
        # Secondary: Balance? Maybe not needed if we just minimize violations.
        # Actually, a simple count of violations is a good heuristic.
        # However, minimizing the *number* of violated sets might not be enough if
        # some sets are "close" to violation.
        # Let's stick to count for simplicity and speed, but use a "soft" energy
        # that also penalizes sets with count 3 (for red) or 9 (for blue) lightly?
        # No, let's just use the hard violation count first. If it stalls, we can add soft penalties.
        # Actually, a better energy is:
        # E = 100 * (num_red_k4 + num_blue_k5) + 1 * (num_red_3 + num_blue_9)
        # This encourages moving away from the edge.
        
        def get_energy(vc_red, vc_blue):
            # vc_red: array of red counts for 4-sets
            # vc_blue: array of blue counts for 5-sets
            hard_v = np.sum(vc_red == 4) + np.sum(vc_blue == 10)
            # Soft penalty: sets that are one flip away from violation
            # Red: count == 3
            # Blue: count == 9
            soft_v = np.sum(vc_red == 3) + np.sum(vc_blue == 9)
            return 1000 * hard_v + soft_v

        current_energy = get_energy(red_count_4, blue_count_5)
        best_energy = current_energy
        
        while time.time() < time_limit:
            # Check time for early exit if no improvement for a while?
            # No, SA needs to run to cool down.
            
            # Pick a random triple to flip
            t_idx = np.random.randint(0, num_triples)
            
            # Calculate delta energy
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_hard = 0
            delta_soft = 0
            
            # Update Red 4-sets
            if old_color == 0: # Was Red, becomes Blue
                # Red count decreases by 1 for all 4-sets containing t_idx
                for s4_idx in triple_to_4sets[t_idx]:
                    c = red_count_4[s4_idx]
                    if c == 4:
                        delta_hard -= 1  # Fixed a violation
                        # c becomes 3, which is a soft violation?
                        # Old: c=4 (hard). New: c=3 (soft).
                        # Change in soft: +1 (3 is soft) - 0 (4 is not soft, it's hard)
                        # Wait, my energy function:
                        # Hard: count==4. Soft: count==3.
                        # Old state: c=4. Energy contribution: 1000*1 + 0 = 1000.
                        # New state: c=3. Energy contribution: 1000*0 + 1 = 1.
                        # Delta = 1 - 1000 = -999.
                        # My delta calculation above was just for hard count.
                        # Let's calculate delta energy directly to be safe.
                        pass
                    elif c == 3:
                        # Old: c=3 (soft). New: c=2 (no violation).
                        # Delta soft: -1.
                        pass
                    # c < 3: no change in violation status.
                
                # Update Blue 5-sets
                # Blue count increases by 1 for all 5-sets containing t_idx
                for s5_idx in triple_to_5sets[t_idx]:
                    c = blue_count_5[s5_idx]
                    if c == 10:
                        # Old: c=10 (hard). New: c=11? No, max is 10.
                        # Wait, if c=10, it's a violation.
                        # If we flip a Red triple to Blue, the blue count increases.
                        # If it was already 10, it stays 10? No, that's impossible.
                        # A 5-set has 10 triples. If blue_count is 10, all are blue.
                        # We are flipping a triple that was Red (0) to Blue (1).
                        # So this triple was NOT blue.
                        # Therefore, blue_count could not be 10.
                        # So c < 10.
                        pass
                    elif c == 9:
                        # Old: c=9 (soft). New: c=10 (hard).
                        # Delta hard: +1. Delta soft: -1.
                        pass
            
            else: # old_color == 1, becomes Red
                # Red count increases by 1
                for s4_idx in triple_to_4sets[t_idx]:
                    c = red_count_4[s4_idx]
                    if c == 4:
                        # Impossible, max 4. If c=4, all are red.
                        # We are flipping Blue to Red. So t_idx was Blue.
                        # So c < 4.
                        pass
                    elif c == 3:
                        # Old: c=3 (soft). New: c=4 (hard).
                        # Delta hard: +1. Delta soft: -1.
                        pass
                
                # Blue count decreases by 1
                for s5_idx in triple_to_5sets[t_idx]:
                    c = blue_count_5[s5_idx]
                    if c == 10:
                        # Old: c=10 (hard). New: c=9 (soft).
                        # Delta hard: -1. Delta soft: +1.
                        pass
                    elif c == 9:
                        # Old: c=9 (soft). New: c=8 (none).
                        # Delta soft: -1.
                        pass

            # This manual delta calculation is error-prone.
            # Given the small n (<=34), num_4sets ~ 46k, num_5sets ~ 324k.
            # triple_to_4sets has ~4 entries. triple_to_5sets has ~6 entries.
            # So updating counts is fast.
            # Let's just apply the flip, calculate new energy, and revert if needed?
            # No, we need to accept/reject based on delta.
            
            # Let's refine the delta calculation properly.
            
            d_energy = 0.0
            
            if old_color == 0: # Red -> Blue
                # Red 4-sets: count decreases
                for s4_idx in triple_to_4sets[t_idx]:
                    c = red_count_4[s4_idx]
                    if c == 4:
                        d_energy -= 999  # 1000 -> 1
                    elif c == 3:
                        d_energy -= 1    # 1 -> 0
                # Blue 5-sets: count increases
                for s5_idx in triple_to_5sets[t_idx]:
                    c = blue_count_5[s5_idx]
                    if c == 9:
                        d_energy += 999  # 1 -> 1000
                    # c < 9: no change in violation status (0 -> 0)
                
            else: # Blue -> Red
                # Red 4-sets: count increases
                for s4_idx in triple_to_4sets[t_idx]:
                    c = red_count_4[s4_idx]
                    if c == 3:
                        d_energy += 999  # 1 -> 1000
                    # c < 3: no change
                # Blue 5-sets: count decreases
                for s5_idx in triple_to_5sets[t_idx]:
                    c = blue_count_5[s5_idx]
                    if c == 10:
                        d_energy -= 999  # 1000 -> 1
                    elif c == 9:
                        d_energy -= 1    # 1 -> 0

            # Acceptance
            if d_energy < 0:
                accept = True
            else:
                # Boltzmann factor
                if T > 0:
                    prob = np.exp(-d_energy / T)
                    accept = np.random.rand() < prob
                else:
                    accept = False
            
            if accept:
                # Apply flip
                colors[t_idx] = new_color
                
                # Update counts
                if old_color == 0:
                    for s4_idx in triple_to_4sets[t_idx]:
                        red_count_4[s4_idx] -= 1
                    for s5_idx in triple_to_5sets[t_idx]:
                        blue_count_5[s5_idx] += 1
                else:
                    for s4_idx in triple_to_4sets[t_idx]:
                        red_count_4[s4_idx] += 1
                    for s5_idx in triple_to_5sets[t_idx]:
                        blue_count_5[s5_idx] -= 1
                
                current_energy += d_energy
                
                # Check for improvement
                # We only care about hard violations for the "best" tracking?
                # Or energy?
                # If energy is lower, it's better.
                # But we must ensure we return a VALID coloring (0 hard violations).
                # So we track best_energy, but only consider it a candidate if hard_violations == 0.
                
                # Calculate hard violations quickly?
                # We can track them incrementally too, or just check periodically.
                # Let's check every 100 steps or if energy drops significantly.
                # Actually, let's just calculate hard violations from counts when we think we might have improved.
                
                # Optimization: Only check for valid solution if current_energy < best_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    # Check if it's a valid solution
                    hard_v = np.sum(red_count_4 == 4) + np.sum(blue_count_5 == 10)
                    if hard_v == 0:
                        best_violations = 0
                        best_colors = colors.copy()
                        best_time = time.time()
                        # If we found a valid solution, we can try to cool down faster?
                        # Or just keep going to ensure stability?
                        # We want to return as soon as we have a valid one?
                        # The task says "Maximize n".
                        # So we try n=34 first. If we find one, we return it.
                        # If we don't find one within time limit, we return None for this n.
                        # So we can return immediately upon finding a valid coloring?
                        # Yes! "A valid colouring on n points certifies R(4,5;3) > n".
                        # So if we find a valid coloring for n, we can return it immediately.
                        return colors.tolist()
            
            # Cooling
            T *= alpha
            if T < T_end:
                T = T_end
            
            # If we are stuck (no improvement in energy for a long time), restart?
            # Simple restart: if no improvement for 5 seconds, reset T and randomize?
            # For now, let's just let it run.
            
        # Time limit reached
        # Check if best_colors is valid
        # We need to verify best_colors
        # Recalculate violations for best_colors
        # This is expensive but we only do it once at the end.
        
        # Actually, we only updated best_colors when hard_v == 0.
        # So if best_violations == 0, best_colors is valid.
        if best_violations == 0:
            return best_colors.tolist()
        return None

    # Try n from 34 downwards
    # We allocate time to each n.
    # Since we return immediately upon finding a valid coloring,
    # we can give each n a fixed chunk of time.
    
    # If we find a valid coloring for n=34, we return.
    # If not, we try n=33, etc.
    
    n_targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
    
    # Time allocation:
    # We have ~118s.
    # Let's give 10s to 34, 10s to 33, 5s to 32, 5s to 31, 5s to 30...
    # Total: 10+10+5*25 = 145. Too much.
    # Let's do:
    # 34: 15s
    # 33: 10s
    # 32: 5s
    # 31: 5s
    # 30: 5s
    # 29: 5s
    # 28: 5s
    # 27: 5s
    # 26: 5s
    # 25: 5s
    # 24: 5s
    # 23: 5s
    # 22: 5s
    # 21: 5s
    # 20: 5s
    # 19: 2s
    # 18: 2s
    # 17: 2s
    # 16: 2s
    # 15: 2s
    # 14: 2s
    # 13: 2s
    # 12: 2s
    # 11: 2s
    # 10: 2s
    # 9: 2s
    # 8: 2s
    # 7: 2s
    # 6: 2s
    # 5: 2s
    # 4: 2s
    
    # Sum: 15+10+10*5+5*5+2*10 = 15+10+50+25+20 = 120.
    # A bit tight. Let's reduce.
    # 34: 10s
    # 33: 5s
    # 32: 5s
    # 31: 5s
    # 30: 5s
    # 29: 5s
    # 28: 5s
    # 27: 5s
    # 26: 5s
    # 25: 5s
    # 24: 5s
    # 23: 5s
    # 22: 5s
    # 21: 5s
    # 20: 5s
    # 19-4: 1s each (16 * 1 = 16s)
    # Sum: 10 + 14*5 + 16 = 10 + 70 + 16 = 96s. Safe.
    
    time_alloc = {
        34: 10.0,
        33: 5.0,
        32: 5.0,
        31: 5.0,
        30: 5.0,
        29: 5.0,
        28: 5.0,
        27: 5.0,
        26: 5.0,
        25: 5.0,
        24: 5.0,
        23: 5.0,
        22: 5.0,
        21: 5.0,
        20: 5.0,
    }
    
    for n in n_targets:
        if time.time() > deadline - 2.0: # Leave 2s for final check
            break
            
        t_limit = time_alloc.get(n, 1.0)
        time_limit = time.time() + t_limit
        if time_limit > deadline - 2.0:
            time_limit = deadline - 2.0
            
        res = solve_sa(n, time_limit)
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback: n=4
    # Valid coloring for n=4:
    # Triples: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    # If we color (0,1,2) Red, others Blue?
    # Red K4: All 4 red. We have 1 red. OK.
    # Blue K5: n=4, no 5-sets. OK.
    # So [0, 1, 1, 1] is valid.
    return {'n': 4, 'colors': [0, 1, 1, 1]}

# Test
if __name__ == "__main__":
    res = construct()
    print(res['n'])
