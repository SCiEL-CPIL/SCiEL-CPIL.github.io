import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin before 150s kill

    def build_structure(n):
        """Precompute all K4 and K5 index tuples and reverse mappings."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))
        
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))
            
        # Reverse mappings
        triple_to_k4s = [[] for _ in range(num_triples)]
        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)
                
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)
            
        return {
            'n': n,
            'num_triples': num_triples,
            'triples': triples,
            'k4s': k4s,
            'k5s': k5s,
            'triple_to_k4s': triple_to_k4s,
            'triple_to_k5s': triple_to_k5s,
            'num_k4': len(k4s),
            'num_k5': len(k5s)
        }

    def count_violations(colors, struct):
        """Count total violations (red K4s + blue K5s)."""
        v4 = 0
        for idxs in struct['k4s']:
            if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                v4 += 1
        v5 = 0
        for idxs in struct['k5s']:
            all_one = True
            for idx in idxs:
                if colors[idx] != 1:
                    all_one = False
                    break
            if all_one:
                v5 += 1
        return v4 + v5

    def delta_violations(t_idx, new_color, colors, struct):
        """Calculate change in violations if triple t_idx is set to new_color."""
        old_color = colors[t_idx]
        if old_color == new_color:
            return 0
            
        d_v = 0
        
        # Update K4s containing t_idx
        for s4_idx in struct['triple_to_k4s'][t_idx]:
            ts = struct['k4s'][s4_idx]
            # Count zeros before
            cnt0 = 0
            for idx in ts:
                if idx == t_idx:
                    if old_color == 0:
                        cnt0 += 1
                else:
                    if colors[idx] == 0:
                        cnt0 += 1
            
            # After flip
            if old_color == 0 and new_color == 1:
                cnt0 -= 1
            elif old_color == 1 and new_color == 0:
                cnt0 += 1
                
            viol_before = 1 if cnt0 == 4 else 0
            viol_after = 1 if cnt0 == 4 else 0
            d_v += (viol_after - viol_before)
            
        # Update K5s containing t_idx
        for s5_idx in struct['triple_to_k5s'][t_idx]:
            ts = struct['k5s'][s5_idx]
            # Count ones before
            cnt1 = 0
            for idx in ts:
                if idx == t_idx:
                    if old_color == 1:
                        cnt1 += 1
                else:
                    if colors[idx] == 1:
                        cnt1 += 1
            
            # After flip
            if old_color == 1 and new_color == 0:
                cnt1 -= 1
            elif old_color == 0 and new_color == 1:
                cnt1 += 1
                
            viol_before = 1 if cnt1 == 10 else 0
            viol_after = 1 if cnt1 == 10 else 0
            d_v += (viol_after - viol_before)
            
        return d_v

    def local_search(n, init_colors, time_limit):
        """
        Perform local search to find a valid coloring for n vertices.
        init_colors: initial coloring (numpy array) or None for random.
        Returns (colors, is_valid).
        """
        struct = build_structure(n)
        num_triples = struct['num_triples']
        
        if init_colors is None:
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        else:
            colors = init_colors.astype(np.int8).copy()
        
        current_cost = count_violations(colors, struct)
        best_colors = colors.copy()
        best_cost = current_cost
        
        if best_cost == 0:
            return best_colors, True
        
        # Simulated annealing with adaptive temperature
        temp = max(1.0, best_cost * 0.1)
        cool_rate = 0.9995
        
        while time.time() < time_limit:
            # Pick random triple
            t = np.random.randint(num_triples)
            
            old_color = colors[t]
            new_color = 1 - old_color
            
            d = delta_violations(t, new_color, colors, struct)
            new_cost = current_cost + d
            
            # Acceptance criterion
            accepted = False
            if d < 0:
                accepted = True
            elif d == 0:
                if np.random.rand() < 0.5:
                    accepted = True
            else:
                if np.random.rand() < np.exp(-d / temp):
                    accepted = True
            
            if accepted:
                colors[t] = new_color
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors, True
            
            temp *= cool_rate
            if temp < 0.05:
                # Reheat
                temp = max(0.5, best_cost * 0.1)
        
        return best_colors, (best_cost == 0)

    # Main strategy: incremental extension
    # Start with a small valid coloring and extend step by step
    
    # First, get a valid coloring for a small n (e.g., n=10)
    # Use a simple heuristic or random search for small n
    
    def find_small_valid(n, time_limit):
        """Find a valid coloring for small n using random restarts."""
        best_colors = None
        best_cost = float('inf')
        
        num_restarts = 20
        steps_per_restart = 5000
        
        for restart in range(num_restarts):
            if time.time() > time_limit:
                break
            
            struct = build_structure(n)
            colors = np.random.randint(0, 2, size=struct['num_triples'], dtype=np.int8)
            current_cost = count_violations(colors, struct)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    return best_colors, True
            
            # Quick hill climbing
            temp = 1.0
            for step in range(steps_per_restart):
                if time.time() > time_limit:
                    break
                
                t = np.random.randint(struct['num_triples'])
                old_color = colors[t]
                new_color = 1 - old_color
                
                d = delta_violations(t, new_color, colors, struct)
                new_cost = current_cost + d
                
                if d < 0:
                    colors[t] = new_color
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return best_colors, True
                elif d == 0 and np.random.rand() < 0.3:
                    colors[t] = new_color
                elif d > 0 and np.random.rand() < np.exp(-d / temp):
                    colors[t] = new_color
                    current_cost = new_cost
                
                temp *= 0.9999
                if temp < 0.1:
                    temp = 0.5
        
        return best_colors, (best_cost == 0)

    # Start with n=10
    current_n = 10
    t_limit_small = start_time + 10.0
    colors, is_valid = find_small_valid(current_n, t_limit_small)
    
    if not is_valid:
        # Fallback: try a few more restarts with more time
        t_limit_small2 = start_time + 20.0
        colors, is_valid = find_small_valid(current_n, t_limit_small2)
    
    if not is_valid:
        # Last resort: return a simple valid coloring for n=9
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': best_colors}
    
    # Now extend from current_n to higher values
    best_n = current_n
    best_final_colors = colors.copy()
    
    # Try to extend up to n=34
    target_n = 34
    
    for new_n in range(current_n + 1, target_n + 1):
        if time.time() > deadline - 10:
            break
        
        # Time allocation: more time for higher n
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 5:
            break
        
        # Allocate time based on how far we are from target
        # More time for higher n
        time_fraction = 0.3 + 0.4 * (new_n - current_n) / (target_n - current_n)
        t_limit = time.time() + remaining_time * time_fraction
        
        # Build structure for new_n
        struct_new = build_structure(new_n)
        num_triples_new = struct_new['num_triples']
        
        # Extend the coloring: map old triples to new indices, randomize new triples
        new_colors = np.zeros(num_triples_new, dtype=np.int8)
        
        # Create mapping from old triple tuples to new indices
        old_triple_to_idx = {t: i for i, t in enumerate(colors)}
        # Actually, we need to map by the triple values
        old_struct = build_structure(current_n)
        old_triple_set = {t: i for i, t in enumerate(old_struct['triples'])}
        
        for i, t in enumerate(struct_new['triples']):
            if t[2] < current_n:  # Old triple
                new_colors[i] = colors[old_triple_set[t]]
            else:
                # New triple involving the new vertex
                # Use a heuristic: color based on the other two vertices
                # Or just random
                new_colors[i] = np.random.randint(0, 2)
        
        # Run local search to fix violations
        colors_new, is_valid_new = local_search(new_n, new_colors, t_limit)
        
        if is_valid_new:
            best_n = new_n
            best_final_colors = colors_new
            current_n = new_n
            colors = colors_new
        else:
            # Try a few more times with different randomizations
            for attempt in range(3):
                if time.time() > t_limit:
                    break
                
                # Re-randomize the new triples
                new_colors2 = new_colors.copy()
                for i, t in enumerate(struct_new['triples']):
                    if t[2] >= current_n:
                        new_colors2[i] = np.random.randint(0, 2)
                
                colors_new, is_valid_new = local_search(new_n, new_colors2, t_limit)
                
                if is_valid_new:
                    best_n = new_n
                    best_final_colors = colors_new
                    current_n = new_n
                    colors = colors_new
                    break
    
    # Final verification
    if best_n >= 10:
        # Verify the coloring is actually valid
        struct_check = build_structure(best_n)
        final_check = count_violations(best_final_colors, struct_check)
        if final_check == 0:
            return {'n': best_n, 'colors': best_final_colors.tolist()}
    
    # Fallback to seed
    n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': best_colors}
