import random
import time
import math
from itertools import combinations

def construct():
    def run_sa(n, time_limit=10.0):
        # Precompute triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        idx_map = {t: i for i, t in enumerate(triples)}
        
        # Precompute 4-sets and their constituent triples
        quads = list(combinations(range(n), 4))
        quad_triples = []
        for q in quads:
            ts = [idx_map[(q[0], q[1], q[2])], idx_map[(q[0], q[1], q[3])], idx_map[(q[0], q[2], q[3])], idx_map[(q[1], q[2], q[3])]]
            quad_triples.append(ts)
            
        # Precompute 5-sets and their constituent triples
        quints = list(combinations(range(n), 5))
        quint_triples = []
        for q in quints:
            ts = []
            for t in combinations(q, 3):
                ts.append(idx_map[t])
            quint_triples.append(ts)
            
        # For fast delta calculation, we need to know for each triple, which quads/quints it belongs to
        # And for each quad/quint, the other triples.
        
        # This might be too slow to build maps for every n if we do it inside the loop.
        # But n is small.
        
        # Initialize colors
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        def get_energy(colors):
            red_viol = 0
            blue_viol = 0
            for ts in quad_triples:
                if colors[ts[0]] == 0 and colors[ts[1]] == 0 and colors[ts[2]] == 0 and colors[ts[3]] == 0:
                    red_viol += 1
            for ts in quint_triples:
                # Check if all 10 are blue (1)
                # Since there are 10 triples, we can sum them. If sum == 10, it's a violation.
                # Or just check one? No, must be all.
                # Optimization: if any is 0, not blue.
                is_blue = True
                for t in ts:
                    if colors[t] != 1:
                        is_blue = False
                        break
                if is_blue:
                    blue_viol += 1
            return red_viol + blue_viol

        # Initial energy
        current_energy = get_energy(colors)
        
        # If already 0, return
        if current_energy == 0:
            return colors

        # Simulated Annealing
        start_time = time.time()
        T = 5.0
        cooling = 0.9999
        min_T = 0.1
        
        # We need to map triple index to the quads and quints it participates in
        # to calculate delta efficiently.
        # triple_to_quads: list of lists of (quad_idx, other_triples_in_quad)
        # triple_to_quints: list of lists of (quint_idx, other_triples_in_quint)
        
        triple_to_quads = [[] for _ in range(num_triples)]
        for q_idx, ts in enumerate(quad_triples):
            for t_idx in ts:
                others = [x for x in ts if x != t_idx]
                triple_to_quads[t_idx].append((q_idx, others))
                
        triple_to_quints = [[] for _ in range(num_triples)]
        for q_idx, ts in enumerate(quint_triples):
            for t_idx in ts:
                others = [x for x in ts if x != t_idx]
                triple_to_quints[t_idx].append((q_idx, others))
        
        while time.time() - start_time < time_limit:
            if T < min_T:
                break
                
            t_idx = random.randint(0, num_triples - 1)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta = 0
            
            # Calculate delta for quads (Red K4)
            for q_idx, others in triple_to_quads[t_idx]:
                # Check if quad was red before
                was_red = True
                if old_color != 0:
                    was_red = False
                else:
                    for o in others:
                        if colors[o] != 0:
                            was_red = False
                            break
                
                # Check if quad is red after
                is_red = True
                if new_color != 0:
                    is_red = False
                else:
                    for o in others:
                        if colors[o] != 0:
                            is_red = False
                            break
                
                if was_red and not is_red:
                    delta -= 1
                elif not was_red and is_red:
                    delta += 1
                    
            # Calculate delta for quints (Blue K5)
            for q_idx, others in triple_to_quints[t_idx]:
                # Check if quint was blue before
                was_blue = True
                if old_color != 1:
                    was_blue = False
                else:
                    for o in others:
                        if colors[o] != 1:
                            was_blue = False
                            break
                
                # Check if quint is blue after
                is_blue = True
                if new_color != 1:
                    is_blue = False
                else:
                    for o in others:
                        if colors[o] != 1:
                            is_blue = False
                            break
                            
                if was_blue and not is_blue:
                    delta -= 1
                elif not was_blue and is_blue:
                    delta += 1
            
            # Acceptance
            if delta <= 0:
                colors[t_idx] = new_color
                current_energy += delta
            else:
                # Metropolis criterion
                # Accept with probability exp(-delta / T)
                if random.random() < math.exp(-delta / T):
                    colors[t_idx] = new_color
                    current_energy += delta
            
            T *= cooling
            
            if current_energy == 0:
                return colors
                
        # Final check
        if current_energy == 0:
            return colors
            
        return None

    # Try to find valid colorings for decreasing n
    for n in [34, 33, 32, 30, 28]:
        res = run_sa(n, time_limit=15.0)
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback
    n = 9
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
