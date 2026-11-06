import itertools
import random
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)

# Precomputation
# iso_third[i][j] = list of k such that (POINTS[i], POINTS[j], POINTS[k]) is isosceles
iso_third = [[[] for _ in range(NUM_POINTS)] for _ in range(NUM_POINTS)]

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def is_isosceles(a, b, c):
    d_ab = dist_sq(a, b)
    d_ac = dist_sq(a, c)
    d_bc = dist_sq(b, c)
    return d_ab == d_ac or d_ab == d_bc or d_ac == d_bc

# Build iso_third
# For each triple, check if isosceles
# Optimization: For each pair (i, j), find all k.
# This is O(N^3) = 256^3 = 16.7M operations. Might be slow in pure Python.
# Let's try to optimize or just do it once at start. 16M simple checks is ~1-2 seconds in Python. Acceptable.

print("Precomputing...")
for i in range(NUM_POINTS):
    pi = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        pj = POINTS[j]
        for k in range(j + 1, NUM_POINTS):
            pk = POINTS[k]
            if is_isosceles(pi, pj, pk):
                # Add k to iso_third[i][j] and iso_third[j][i]
                # Also add i to iso_third[j][k] and iso_third[k][j]
                # And add j to iso_third[i][k] and iso_third[k][i]
                
                # We need symmetric access for updates.
                # iso_third[a][b] should contain c if {a,b,c} is isosceles.
                
                # Pair (i, j) is "blocked" by k? No, k is the third point.
                # If we add k, it forms isosceles with (i, j).
                # So if we track conflict_count for q, and we add p, 
                # we need to know which existing pairs in S form isosceles with p.
                # That is: for each r in S, if (p, r) is a pair, which q's form isosceles?
                # Wait.
                # conflict_count[q] = number of pairs {a, b} in S such that {q, a, b} is isosceles.
                # If we add p to S:
                # 1. New isosceles triangles involving p:
                #    These are triples {p, a, b} where a, b in S_old.
                #    Count = number of pairs {a, b} in S_old such that {p, a, b} is isosceles.
                #    This is exactly conflict_count[p] BEFORE adding p.
                # 2. Update conflict_count for q not in S_new:
                #    For each q not in S_new:
                #    conflict_count[q] increases by number of pairs {a, b} in S_new \ S_old = {p} such that {q, a, b} is isosceles.
                #    The pairs involving p are {p, r} where r in S_old.
                #    So for each r in S_old:
                #       For each q such that {q, p, r} is isosceles:
                #           conflict_count[q] += 1
                #    We need to find q given p, r.
                #    This is exactly iso_third[p][r] (or iso_third[r][p]).
                
                # So we need iso_third[p][r] to list all q such that {p, r, q} is isosceles.
                
                # Let's populate iso_third[i][j] with k such that {i, j, k} is isosceles.
                
                iso_third[i][j].append(k)
                iso_third[j][i].append(k)
                
                iso_third[j][k].append(i)
                iso_third[k][j].append(i)
                
                iso_third[i][k].append(j)
                iso_third[k][i].append(j)

print("Precomputation done.")

def construct():
    # Simulated Annealing
    # State: S (set of indices), conflict_count (list of length NUM_POINTS), total_isos, num_addable
    # We maintain conflict_count for all points, but only count q not in S for num_addable.
    
    best_set = None
    best_size = float('inf')
    
    # Initialize
    S = set()
    conflict_count = [0] * NUM_POINTS
    total_isos = 0
    num_addable = NUM_POINTS # All points can be added to empty set
    
    # To speed up, we can start with a greedy construction
    # Greedy: Add points that block the most others?
    # Or just random start.
    
    # Let's do a few restarts
    for restart in range(10):
        # Random initial set? Or empty?
        # Empty is safe (total_isos=0).
        S = set()
        conflict_count = [0] * NUM_POINTS
        total_isos = 0
        num_addable = NUM_POINTS
        
        T = 100.0
        T_min = 0.01
        cooling = 0.995
        
        current_energy = 10 * num_addable + len(S) # total_isos is 0
        
        # SA Loop
        for step in range(5000):
            if T < T_min:
                break
            
            # Choose a random point to flip
            p = random.randint(0, NUM_POINTS - 1)
            
            if p in S:
                # Remove p
                # 1. total_isos decreases by conflict_count[p] (pairs in S\{p} that form isosceles with p)
                #    Note: conflict_count[p] is currently calculated w.r.t S.
                #    Wait, conflict_count[p] counts pairs {a,b} in S such that {p,a,b} is isosceles.
                #    If we remove p, these triangles disappear.
                #    So new_total_isos = total_isos - conflict_count[p]
                old_conf_p = conflict_count[p]
                total_isos -= old_conf_p
                
                # 2. Update conflict_count for q not in S (after removal, S_new = S\{p})
                #    For each q not in S_new:
                #    conflict_count[q] decreases by number of pairs {a,b} in S_new such that {q,a,b} is isosceles and one of a,b is p.
                #    Pairs involving p in S_old are {p, r} for r in S_old\{p}.
                #    So for each r in S_old\{p}:
                #       For each q in iso_third[p][r]:
                #           if q not in S_new:
                #               conflict_count[q] -= 1
                #           # If q in S_new, we don't care for num_addable, but we update the count for correctness if q is re-added later?
                #           # Yes, we should update all q to keep conflict_count accurate.
                #           # But we only use q not in S for num_addable.
                
                S_old = S.copy()
                S.remove(p)
                
                for r in S: # r in S_new
                    for q in iso_third[p][r]:
                        conflict_count[q] -= 1
                        # Update num_addable if q was addable and became non-addable or vice versa?
                        # We'll recalculate num_addable or update incrementally.
                        # Incremental is tricky. Let's just track num_addable.
                        # Before: q not in S_old.
                        # After: q not in S_new (since p was removed, S_new is subset of S_old, so if q not in S_old, q not in S_new).
                        # So q is still a candidate for addable.
                        # If conflict_count[q] was 0 and becomes 1, num_addable decreases.
                        # If conflict_count[q] was 1 and becomes 0, num_addable increases.
                        
                # Recalculate num_addable is O(N). N=256. 5000 steps * 10 restarts * 256 = 12.8M ops. Fast.
                num_addable = sum(1 for q in range(NUM_POINTS) if q not in S and conflict_count[q] == 0)
                
                new_energy = 1000 * total_isos + 10 * num_addable + len(S)
                delta_E = new_energy - current_energy
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    current_energy = new_energy
                    # Accept move
                else:
                    # Reject move: Restore state
                    S.add(p)
                    total_isos += old_conf_p
                    for r in S: # r in S_old
                        # Only r != p
                        if r != p:
                            for q in iso_third[p][r]:
                                conflict_count[q] += 1
                    num_addable = sum(1 for q in range(NUM_POINTS) if q not in S and conflict_count[q] == 0)
                    # Note: We don't update current_energy if rejected
            
            else:
                # Add p
                # 1. total_isos increases by conflict_count[p]
                old_conf_p = conflict_count[p]
                total_isos += old_conf_p
                
                S.add(p)
                
                # 2. Update conflict_count for q not in S (after addition, S_new = S)
                #    For each q not in S:
                #    conflict_count[q] increases by number of pairs {a,b} in S_new such that {q,a,b} is isosceles and one of a,b is p.
                #    Pairs involving p in S_new are {p, r} for r in S_old = S\{p}.
                #    So for each r in S\{p}:
                #       For each q in iso_third[p][r]:
                #           conflict_count[q] += 1
                
                for r in S:
                    if r != p:
                        for q in iso_third[p][r]:
                            conflict_count[q] += 1
                
                num_addable = sum(1 for q in range(NUM_POINTS) if q not in S and conflict_count[q] == 0)
                
                new_energy = 1000 * total_isos + 10 * num_addable + len(S)
                delta_E = new_energy - current_energy
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    current_energy = new_energy
                    # Accept move
                else:
                    # Reject move
                    S.remove(p)
                    total_isos -= old_conf_p
                    for r in S:
                        if r != p:
                            for q in iso_third[p][r]:
                                conflict_count[q] -= 1
                    num_addable = sum(1 for q in range(NUM_POINTS) if q not in S and conflict_count[q] == 0)
            
            T *= cooling
            
            # Check if we have a valid solution
            if total_isos == 0 and num_addable == 0:
                if len(S) < best_size:
                    best_size = len(S)
                    best_set = S.copy()
                # We can continue to try to minimize size, or restart
                # Let's continue SA to see if we can find smaller one
            
    if best_set is None:
        # Fallback to greedy if SA failed (unlikely)
        S = []
        for p in POINTS:
            ok = True
            for a, b in itertools.combinations(S, 2):
                if is_isosceles(p, a, b):
                    ok = False
                    break
            if ok:
                S.append(p)
        return S
        
    return [POINTS[i] for i in best_set]
