import time
import random
import math
from itertools import combinations

_structure_cache = {}

def get_structure(n):
    if n in _structure_cache:
        return _structure_cache[n]
        
    edges = list(combinations(range(n), 3))
    num_edges = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    edge_to_k4s = [[] for _ in range(num_edges)]
    edge_to_k5s = [[] for _ in range(num_edges)]
    
    k4_edges = []
    for k4 in k4s:
        a, b, c, d = k4
        e1 = edge_index[(a,b,c)]
        e2 = edge_index[(a,b,d)]
        e3 = edge_index[(a,c,d)]
        e4 = edge_index[(b,c,d)]
        k4_edges.append([e1, e2, e3, e4])
        for e in [e1, e2, e3, e4]:
            edge_to_k4s[e].append(len(k4_edges) - 1)
            
    k5_edges = []
    for k5 in k5s:
        es = [edge_index[e] for e in combinations(k5, 3)]
        k5_edges.append(es)
        for e in es:
            edge_to_k5s[e].append(len(k5_edges) - 1)
            
    _structure_cache[n] = {
        'num_edges': num_edges,
        'edge_to_k4s': edge_to_k4s,
        'edge_to_k5s': edge_to_k5s,
        'k4_edges': k4_edges,
        'k5_edges': k5_edges,
        'num_k4s': len(k4s),
        'num_k5s': len(k5s)
    }
    return _structure_cache[n]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # Try n=34 primarily
    # We will use multiple restarts. Each restart gets a slice of time.
    # If we find a solution for 34, we return immediately.
    
    # Time allocation:
    # Spend most time on 34.
    # If 34 fails, try 33, then 32.
    
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 5:
            break
        
        if n == 34:
            max_time_for_n = 50
        elif n == 33:
            max_time_for_n = 20
        elif n == 32:
            max_time_for_n = 10
        else:
            max_time_for_n = 5
            
        found = False
        start_n_time = time.time()
        
        # Multiple restarts
        while time.time() - start_n_time < max_time_for_n:
            if time.time() > deadline - 5:
                break
                
            # Each attempt gets a slice of the remaining time for this n
            # Shorter runs are often better for SA with restarts
            time_limit = 3.0 
            if time.time() + time_limit > deadline - 5:
                 time_limit = max(1.0, deadline - 5 - time.time())
                 
            result = solve_fast_targeted(n, time_limit)
            if result is not None:
                best_n = n
                best_colors = result
                found = True
                break
        
        if found:
            break
            
    if best_n == 0:
        # Fallback to a known small solution
        # n=9 is always valid with simple modular coloring?
        # Let's check: n=9.
        # Actually, just return a small valid one if we fail.
        # For n=1..8, it's trivial.
        n = 8
        # Random coloring for small n is very likely to be valid or we can brute force
        # But we need a valid one.
        # Let's use the modular one from seed, which worked for n=9?
        # Seed used: 0 if (a+b+c)%3==0 else 1
        # Let's verify if that is valid for n=9.
        # If not, we might return invalid.
        # Better to return a trivially valid one: all red? No, red K4 forbidden.
        # All blue? No, blue K5 forbidden.
        # For n < 4, no K4. For n < 5, no K5.
        # So n=4 is safe if no red K4. If we color all blue, no red K4. No blue K5 (n<5).
        # So n=4 with all blue is valid.
        # Let's try to find a valid one for n=8 quickly if we fail 30-34.
        # But the loop above tries down to 30.
        # If we fail all, we return n=8 with a safe coloring.
        
        # Safe coloring for n=8:
        # We need to avoid Red K4 and Blue K5.
        # If we color all edges Blue:
        # No Red K4 (0 red edges).
        # Blue K5? We have 5 vertices. Are all 10 edges blue? Yes.
        # So all blue is INVALID for n>=5.
        
        # If we color all edges Red:
        # Red K4? We have 4 vertices. Are all 4 edges red? Yes.
        # So all red is INVALID for n>=4.
        
        # We need a mix.
        # For n=8, let's just run a quick solver with a very short time limit.
        # Or use a known small construction.
        # Actually, the fallback in the seed was n=9.
        # Let's assume the solver for n=30+ will succeed.
        # If not, we return n=1 (trivially valid).
        # But score would be low.
        # Let's try n=8 with a quick targeted SA.
        
        if time.time() < deadline - 2:
             res = solve_fast_targeted(8, 2.0)
             if res:
                 return {'n': 8, 'colors': res}
        
        # Absolute fallback: n=1
        return {'n': 1, 'colors': [0]}
        
    return {'n': best_n, 'colors': best_colors}

def solve_fast_targeted(n, time_limit=5):
    """
    Targeted Simulated Annealing.
    Focuses search on edges involved in violations.
    """
    start_time = time.time()
    
    struct = get_structure(n)
    num_edges = struct['num_edges']
    edge_to_k4s = struct['edge_to_k4s']
    edge_to_k5s = struct['edge_to_k5s']
    k4_edges = struct['k4_edges']
    k5_edges = struct['k5_edges']
    num_k4s = struct['num_k4s']
    num_k5s = struct['num_k5s']
    
    # Initial Random Coloring
    # Bias towards Red (0) to make Blue K5s less likely?
    # Or uniform?
    # Let's try uniform first, or slight bias.
    # If p_red = 0.5, it's symmetric.
    # Blue K5 requires 10 edges. Red K4 requires 4 edges.
    # It is harder to form a Blue K5 by chance if Blue is less frequent?
    # Actually, if we want to AVOID Blue K5, we want Blue to be rare.
    # So p_red = 0.7 might be good.
    p_red = 0.7
    colors = [0 if random.random() < p_red else 1 for _ in range(num_edges)]
    
    # Calculate initial counts
    red_counts = [0] * num_k4s
    blue_counts = [0] * num_k5s
    
    for i, e_list in enumerate(k4_edges):
        for e in e_list:
            if colors[e] == 0:
                red_counts[i] += 1
                
    for i, e_list in enumerate(k5_edges):
        for e in e_list:
            if colors[e] == 1:
                blue_counts[i] += 1
                
    total_violations = 0
    for rc in red_counts:
        if rc == 4:
            total_violations += 1
    for bc in blue_counts:
        if bc == 10:
            total_violations += 1
            
    if total_violations == 0:
        return colors.copy()
        
    best_colors = colors.copy()
    best_violations = total_violations
    
    # Targeted SA
    T = 1.0
    cooling = 0.995
    exp = math.exp
    rand = random.random
    randint = random.randint
    
    # Track edges that are in violations to target them
    # We will rebuild this list periodically or maintain it.
    # Rebuilding is O(E * avg_clique_size) which is okay for n=34.
    # num_edges ~ 5984.
    # Rebuilding every 100 steps is fine.
    
    steps_since_rebuild = 0
    bad_edges = []
    
    def rebuild_bad_edges():
        bad_set = set()
        for i, rc in enumerate(red_counts):
            if rc == 4:
                for e in k4_edges[i]:
                    bad_set.add(e)
        for i, bc in enumerate(blue_counts):
            if bc == 10:
                for e in k5_edges[i]:
                    bad_set.add(e)
        return list(bad_set)
        
    bad_edges = rebuild_bad_edges()
    
    while time.time() < start_time + time_limit:
        # Pick an edge
        if bad_edges and rand() < 0.8:
            # Pick from bad edges
            e_idx = bad_edges[randint(0, len(bad_edges) - 1)]
        else:
            # Pick random
            e_idx = randint(0, num_edges - 1)
            
        old_color = colors[e_idx]
        new_color = 1 - old_color
        
        delta_v = 0
        
        # Update K4s
        for k4_idx in edge_to_k4s[e_idx]:
            if old_color == 0:
                # Red -> Blue
                red_counts[k4_idx] -= 1
                if red_counts[k4_idx] == 3: # Was 4, now 3 (violation removed)
                    delta_v -= 1
            else:
                # Blue -> Red
                red_counts[k4_idx] += 1
                if red_counts[k4_idx] == 4: # Was 3, now 4 (violation added)
                    delta_v += 1
                    
        # Update K5s
        for k5_idx in edge_to_k5s[e_idx]:
            if old_color == 1:
                # Blue -> Red
                blue_counts[k5_idx] -= 1
                if blue_counts[k5_idx] == 9: # Was 10, now 9 (violation removed)
                    delta_v -= 1
            else:
                # Red -> Blue
                blue_counts[k5_idx] += 1
                if blue_counts[k5_idx] == 10: # Was 9, now 10 (violation added)
                    delta_v += 1
                    
        new_v = total_violations + delta_v
        
        # Acceptance
        accepted = False
        if delta_v <= 0:
            accepted = True
        else:
            if T > 0.01:
                if rand() < exp(-delta_v / T):
                    accepted = True
                    
        if accepted:
            colors[e_idx] = new_color
            total_violations = new_v
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                
            if total_violations == 0:
                return colors.copy()
                
        else:
            # Revert counts
            for k4_idx in edge_to_k4s[e_idx]:
                if old_color == 0:
                    red_counts[k4_idx] += 1
                else:
                    red_counts[k4_idx] -= 1
            for k5_idx in edge_to_k5s[e_idx]:
                if old_color == 1:
                    blue_counts[k5_idx] += 1
                else:
                    blue_counts[k5_idx] -= 1
                    
        T *= cooling
        
        # Periodically rebuild bad edges list
        steps_since_rebuild += 1
        if steps_since_rebuild > 200:
            bad_edges = rebuild_bad_edges()
            steps_since_rebuild = 0
            
    if best_violations == 0:
        return best_colors
    return None
