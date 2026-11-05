import numpy as np
import time
import random
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    def build_structure(n):
        """Precompute all structures needed for fast cost evaluation."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # K4 structures (4 vertices, 4 triples each)
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))

        # K5 structures (5 vertices, 10 triples each)
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))

        # Reverse indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]

        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)

        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        return {
            'triples': triples,
            'num_triples': num_triples,
            'k4s': k4s,
            'k5s': k5s,
            'triple_to_k4s': triple_to_k4s,
            'triple_to_k5s': triple_to_k5s
        }

    def compute_cost(colors, struct):
        """Compute total number of violations."""
        red_count = 0
        for idxs in struct['k4s']:
            if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                red_count += 1
        blue_count = 0
        for idxs in struct['k5s']:
            if all(colors[i] == 1 for i in idxs):
                blue_count += 1
        return red_count + blue_count

    def get_delta_cost(t, new_color, colors, struct):
        """Calculate change in cost if triple t is flipped to new_color."""
        old_color = colors[t]
        if old_color == new_color:
            return 0

        delta = 0

        # Check K4s containing t
        for k4_idx in struct['triple_to_k4s'][t]:
            idxs = struct['k4s'][k4_idx]
            red_before = sum(1 for i in idxs if colors[i] == 0)
            if old_color == 0:
                red_after = red_before - 1
            else:
                red_after = red_before + 1

            if red_before == 4 and red_after < 4:
                delta -= 1
            elif red_before < 4 and red_after == 4:
                delta += 1

        # Check K5s containing t
        for k5_idx in struct['triple_to_k5s'][t]:
            idxs = struct['k5s'][k5_idx]
            blue_before = sum(1 for i in idxs if colors[i] == 1)
            if old_color == 1:
                blue_after = blue_before - 1
            else:
                blue_after = blue_before + 1

            if blue_before == 10 and blue_after < 10:
                delta -= 1
            elif blue_before < 10 and blue_after == 10:
                delta += 1

        return delta

    def greedy_local_search(n, time_limit, seed=0, init_colors=None):
        """Greedy local search: flip triple in most conflicting violation."""
        random.seed(seed)
        np.random.seed(seed)

        struct = build_structure(n)
        num_triples = struct['num_triples']

        if init_colors is not None:
            colors = init_colors.copy()
        else:
            colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        
        current_cost = compute_cost(colors, struct)

        if current_cost == 0:
            return colors, True

        best_cost = current_cost
        best_colors = colors.copy()

        max_iterations = 50000
        iteration = 0

        while time.time() < time_limit and iteration < max_iterations:
            iteration += 1

            # Find a random violation
            k4s = struct['k4s']
            k5s = struct['k5s']

            red_violations = []
            for i, idxs in enumerate(k4s):
                if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                    red_violations.append(i)

            blue_violations = []
            for i, idxs in enumerate(k5s):
                if all(colors[j] == 1 for j in idxs):
                    blue_violations.append(i)

            if not red_violations and not blue_violations:
                break

            # Prefer fixing red K4s (2x weight)
            if red_violations and (not blue_violations or random.random() < 0.6):
                k4_idx = random.choice(red_violations)
                idxs = k4s[k4_idx]
            else:
                k5_idx = random.choice(blue_violations)
                idxs = k5s[k5_idx]

            # Try flipping each triple in the violation
            best_flip = None
            best_delta = 0

            for t in idxs:
                new_color = 1 - colors[t]
                d = get_delta_cost(t, new_color, colors, struct)
                if d < best_delta or (d == best_delta and random.random() < 0.5):
                    best_delta = d
                    best_flip = t

            if best_flip is not None:
                new_color = 1 - colors[best_flip]
                colors[best_flip] = new_color
                current_cost += best_delta

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()

                    if best_cost == 0:
                        return best_colors, True
            else:
                # No improving flip found; do a random perturbation
                t = random.randint(0, num_triples - 1)
                colors[t] = 1 - colors[t]
                current_cost = compute_cost(colors, struct)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()

        return best_colors, (best_cost == 0)

    def verify_coloring(n, colors):
        """Verify that the coloring has no red K4 and no blue K5."""
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a, b, c)]] == 0 and
                colors[triple_to_idx[(a, b, d)]] == 0 and
                colors[triple_to_idx[(a, c, d)]] == 0 and
                colors[triple_to_idx[(b, c, d)]] == 0):
                return False

        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False

        return True

    def incremental_backtracking(n, time_limit, seed=0):
        """
        Attempt to build a valid coloring for n vertices by adding the last vertex
        to a valid coloring of n-1 vertices using backtracking.
        """
        if n < 10:
            return None, False
            
        # We need a valid coloring for n-1. We'll try to find one quickly or use a random one if we have time.
        # For simplicity in this integrated function, we assume we are called when we have a candidate for n-1 
        # or we generate one. 
        # Actually, let's just use this as a *refiner* or *constructor* from scratch for small n, 
        # but for large n, we need a base.
        
        # Let's try to construct from n-1.
        # This function is called in a loop for increasing n.
        # We maintain `current_valid_colors` and `current_valid_n`.
        
        # If we don't have a valid base, return None.
        # The main loop will handle the base case.
        
        # For now, this function is a placeholder for the logic that would extend a base.
        # The actual extension logic is complex. Let's rely on the local search 
        # initialized with a "greedy extension" heuristic.
        
        return None, False

    def greedy_extension_init(n, base_colors, base_n):
        """
        Given a valid coloring for base_n, create an initial coloring for n
        by greedily assigning colors to new triples involving vertices base_n..n-1.
        """
        if base_n != n - 1:
            return None
            
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        colors = [None] * len(triples)
        
        # Copy base colors
        for t in combinations(range(base_n), 3):
            colors[triple_to_idx[t]] = base_colors[triple_to_idx[t]]
            
        # Precompute structures for base_n to check constraints
        # We need to check if adding a new triple creates a violation.
        
        # New triples are those involving at least one vertex >= base_n.
        # Since we add only one vertex (v = n-1), new triples are (n-1, a, b) with a < b < n-1.
        v = n - 1
        new_triples = []
        for a in range(v):
            for b in range(a+1, v):
                t = (v, a, b)
                new_triples.append(triple_to_idx[t])
                
        # We need to check constraints.
        # Red K4 violation: If we color (v,a,b) Red, and there exists c < v such that (a,b,c) is Red, 
        # and we also color (v,a,c) and (v,b,c) Red, we have a Red K4 on {v,a,b,c}.
        # This is a global constraint. Simple greedy might fail.
        
        # Heuristic: Randomly assign, but bias towards Red if it doesn't immediately complete a Red K4 
        # with already colored triples.
        
        # For simplicity, let's just randomize the new triples.
        # The local search will fix it.
        for t_idx in new_triples:
            colors[t_idx] = random.randint(0, 1)
            
        return np.array(colors, dtype=np.int8)

    # Start with a known small valid coloring
    final_n = 9
    triples_9 = list(combinations(range(9), 3))
    final_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_9]
    
    current_valid_n = 9
    current_valid_colors = final_colors

    # Try to find valid colorings for larger n
    target_ns = list(range(10, 35))

    for n in target_ns:
        if time.time() > deadline - 3:
            break

        # Allocate time based on n
        if n <= 15:
            time_budget = min(2.0, (deadline - time.time()) * 0.25)
        elif n <= 25:
            time_budget = min(4.0, (deadline - time.time()) * 0.3)
        else:
            time_budget = min(6.0, (deadline - time.time()) * 0.35)

        if time_budget < 0.5:
            break

        time_limit = time.time() + time_budget
        found = False

        # Strategy 1: Greedy Extension + Local Search
        # Initialize using the previous valid coloring
        init_colors = greedy_extension_init(n, current_valid_colors, current_valid_n)
        
        if init_colors is not None:
            # Run local search with this initialization
            for seed in range(3):
                if time.time() > time_limit:
                    break
                colors, is_valid = greedy_local_search(n, time_limit, seed=seed, init_colors=init_colors)
                if is_valid:
                    if verify_coloring(n, colors):
                        final_n = n
                        final_colors = colors.tolist()
                        current_valid_n = n
                        current_valid_colors = colors
                        found = True
                        break
        else:
            # Fallback to random init if extension failed (shouldn't happen if n-1 was valid)
            for seed in range(3):
                if time.time() > time_limit:
                    break
                colors, is_valid = greedy_local_search(n, time_limit, seed=seed)
                if is_valid:
                    if verify_coloring(n, colors):
                        final_n = n
                        final_colors = colors.tolist()
                        current_valid_n = n
                        current_valid_colors = colors
                        found = True
                        break
                        
        if found:
            continue
            
        # If not found, try a few more random restarts with pure random init
        # Just in case the greedy extension got stuck in a bad local minimum
        if time.time() < time_limit - 1:
             for seed in range(2, 5):
                if time.time() > time_limit:
                    break
                colors, is_valid = greedy_local_search(n, time_limit, seed=seed)
                if is_valid:
                    if verify_coloring(n, colors):
                        final_n = n
                        final_colors = colors.tolist()
                        current_valid_n = n
                        current_valid_colors = colors
                        found = True
                        break

    # Final verification
    if not verify_coloring(final_n, final_colors):
        final_n = 9
        triples_9 = list(combinations(range(9), 3))
        final_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_9]

    return {'n': final_n, 'colors': final_colors}

if __name__ == "__main__":
    result = construct()
    print(f"Found valid coloring for n={result['n']}")
