"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 145
    
    def make_index_map(n):
        """Map triple (a,b,c) to its lexicographic index."""
        idx = 0
        triple_to_idx = {}
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_to_idx[(a, b, c)] = idx
                    idx += 1
        return triple_to_idx
    
    def get_4sets(n):
        """Return list of (quad, [list of 4 triple indices])."""
        result = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            result.append((quad, triples))
        return result
    
    def get_5sets(n):
        """Return list of (quint, [list of 10 triple indices])."""
        result = []
        for quint in combinations(range(n), 5):
            triples = list(combinations(quint, 3))
            result.append((quint, triples))
        return result
    
    # Try n values from high to low
    for n in range(34, 8, -1):
        if time.time() > deadline:
            break
        
        triple_to_idx = make_index_map(n)
        n_edges = len(triple_to_idx)
        
        # Precompute 4-sets and 5-sets
        four_sets = get_4sets(n)
        five_sets = get_5sets(n)
        
        # For each edge, precompute which 4-sets and 5-sets it's in
        edge_in_4sets = [[] for _ in range(n_edges)]
        edge_in_5sets = [[] for _ in range(n_edges)]
        
        for i, (quad, triples) in enumerate(four_sets):
            for t in triples:
                edge_in_4sets[triple_to_idx[t]].append(i)
        
        for i, (quint, triples) in enumerate(five_sets):
            for t in triples:
                edge_in_5sets[triple_to_idx[t]].append(i)
        
        # Try multiple random initial colorings
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initial coloring
            colors = [random.choice([0, 1]) for _ in range(n_edges)]
            
            # Compute initial violation counts
            # red_count[i] = number of red edges in 4-set i
            red_count = [0] * len(four_sets)
            for i, (quad, triples) in enumerate(four_sets):
                for t in triples:
                    if colors[triple_to_idx[t]] == 0:
                        red_count[i] += 1
            
            # blue_count[j] = number of blue edges in 5-set j
            blue_count = [0] * len(five_sets)
            for j, (quint, triples) in enumerate(five_sets):
                for t in triples:
                    if colors[triple_to_idx[t]] == 1:
                        blue_count[j] += 1
            
            # Count violations
            red_violations = sum(1 for rc in red_count if rc == 4)
            blue_violations = sum(1 for bc in blue_count if bc == 10)
            
            if red_violations == 0 and blue_violations == 0:
                return {'n': n, 'colors': colors}
            
            # Local search
            max_iters = 20000
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                if red_violations == 0 and blue_violations == 0:
                    return {'n': n, 'colors': colors}
                
                # Pick a violating edge to flip
                # Prefer edges in violating sets
                flip_candidates = []
                
                if red_violations > 0:
                    # Find a violating 4-set
                    for i in range(len(four_sets)):
                        if red_count[i] == 4:
                            # Pick a random edge in this 4-set
                            quad, triples = four_sets[i]
                            t = random.choice(triples)
                            flip_candidates.append(triple_to_idx[t])
                            break
                
                if blue_violations > 0:
                    # Find a violating 5-set
                    for j in range(len(five_sets)):
                        if blue_count[j] == 10:
                            # Pick a random edge in this 5-set
                            quint, triples = five_sets[j]
                            t = random.choice(triples)
                            flip_candidates.append(triple_to_idx[t])
                            break
                
                if not flip_candidates:
                    # No violating sets found (shouldn't happen), pick random
                    flip_candidates.append(random.randint(0, n_edges - 1))
                
                edge_idx = random.choice(flip_candidates)
                
                # Flip the edge
                old_color = colors[edge_idx]
                new_color = 1 - old_color
                colors[edge_idx] = new_color
                
                # Update red_count for 4-sets containing this edge
                for i in edge_in_4sets[edge_idx]:
                    if old_color == 0:
                        red_count[i] -= 1
                        if red_count[i] == 3:
                            red_violations -= 1  # was 4, now 3
                    else:
                        red_count[i] += 1
                        if red_count[i] == 4:
                            red_violations += 1  # was 3, now 4
                
                # Update blue_count for 5-sets containing this edge
                for j in edge_in_5sets[edge_idx]:
                    if old_color == 1:
                        blue_count[j] -= 1
                        if blue_count[j] == 9:
                            blue_violations -= 1  # was 10, now 9
                    else:
                        blue_count[j] += 1
                        if blue_count[j] == 10:
                            blue_violations += 1  # was 9, now 10
        
        # If we get here, n wasn't achieved, try smaller n
    
    # Fallback: return a small valid coloring
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
