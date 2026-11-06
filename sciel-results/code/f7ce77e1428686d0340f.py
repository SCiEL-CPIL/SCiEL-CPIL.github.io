import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Precompute modular arithmetic tables for GF(2)^6
    # We'll use 6-bit integers (0..63) as vertices
    # XOR is addition in GF(2)^6
    
    def xor(a, b):
        return a ^ b
    
    def dot_product(a, b):
        """Dot product over GF(2)"""
        x = a & b
        count = 0
        while x:
            count += 1
            x &= x - 1
        return count % 2
    
    # Precompute dot products for all pairs
    dot_table = [[0] * 64 for _ in range(64)]
    for i in range(64):
        for j in range(64):
            dot_table[i][j] = dot_product(i, j)
    
    # Precompute all triples and their K4/K5 incidences for n=34
    # We'll use a random ordering of 34 vertices from the 64 available
    # But first, let's try the algebraic construction on all 64 vertices
    # and see if we can find a valid subset of 34.
    
    # Actually, let's think about this differently.
    # The algebraic construction gives a coloring on 64 vertices.
    # We need to find 34 vertices that form a valid sub-hypergraph.
    # Or we can use the algebraic coloring as a heuristic for local search.
    
    # Let's try a different approach: use the algebraic structure to generate
    # a good initial coloring, then use local search to refine it.
    
    def algebraic_color(a, b, c):
        """
        Color triple (a,b,c) based on algebraic properties.
        Returns 0 (red) or 1 (blue).
        """
        # Use the fact that in GF(2)^6, the determinant of (a,b,c) is 0
        # if they are linearly dependent.
        # For 3 vectors in GF(2)^6, they are linearly dependent if
        # a + b + c = 0, i.e., a = b + c (XOR).
        # But we want a more nuanced coloring.
        
        # Let's use: color = 0 if a·b + b·c + c·a = 0, else 1
        # This is a quadratic form.
        val = (dot_table[a][b] + dot_table[b][c] + dot_table[c][a]) % 2
        return val
    
    def solve_n(n, seed=42):
        """
        Attempts to find a valid coloring for n vertices.
        Uses algebraic initial coloring + local search.
        """
        if n < 1:
            return None
        
        random.seed(seed)
        
        # Generate all triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Generate K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # Precompute incidence lists
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        k4_triples = []
        for k4 in k4s:
            t_indices = [triple_to_idx[t] for t in combinations(k4, 3)]
            k4_triples.append(t_indices)
            for idx in t_indices:
                triple_to_k4s[idx].append(len(k4_triples) - 1)
                
        k5_triples = []
        for k5 in k5s:
            t_indices = [triple_to_idx[t] for t in combinations(k5, 3)]
            k5_triples.append(t_indices)
            for idx in t_indices:
                triple_to_k5s[idx].append(len(k5_triples) - 1)
                
        num_k4s = len(k4s)
        num_k5s = len(k5s)
        
        # Try multiple algebraic colorings with different vertex orderings
        best_result = None
        max_restarts = 10
        restart_time_limit = 8.0  # seconds per restart
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Create a random permutation of vertices to break symmetry
            perm = list(range(n))
            random.shuffle(perm)
            inv_perm = [0] * n
            for i, p in enumerate(perm):
                inv_perm[p] = i
            
            # Initial coloring using algebraic heuristic
            colors = [0] * num_triples
            for i, (a, b, c) in enumerate(triples):
                # Map to permuted indices
                pa, pb, pc = perm[a], perm[b], perm[c]
                colors[i] = algebraic_color(pa, pb, pc)
            
            # Initialize red counts
            rc_k4 = [0] * num_k4s
            rc_k5 = [0] * num_k5s
            
            for i, c in enumerate(colors):
                if c == 0:  # Red
                    for k4_idx in triple_to_k4s[i]:
                        rc_k4[k4_idx] += 1
                    for k5_idx in triple_to_k5s[i]:
                        rc_k5[k5_idx] += 1
                        
            # Count initial violations
            violations = 0
            for r in rc_k4:
                if r == 4:
                    violations += 1
            for r in rc_k5:
                if r == 0:
                    violations += 1
                    
            if violations == 0:
                return colors
            
            # Greedy phase: eliminate violations
            max_greedy_steps = 5000
            for step in range(max_greedy_steps):
                if time.time() > deadline - 5:
                    break
                if violations == 0:
                    return colors
                
                # Find a violated K4 or K5 and fix it
                fixed = False
                for _ in range(20):  # Try random fixes
                    # Pick a random violated K4 or K5
                    if random.random() < 0.5 and violations > 0:
                        # Try to fix a red K4
                        violated_k4s = [i for i, r in enumerate(rc_k4) if r == 4]
                        if violated_k4s:
                            k4_idx = random.choice(violated_k4s)
                            t_indices = k4_triples[k4_idx]
                            # Flip one random triple in this K4 to blue
                            t = random.choice(t_indices)
                            if colors[t] == 0:  # It's red
                                # Flip to blue
                                colors[t] = 1
                                for k4_i in triple_to_k4s[t]:
                                    rc_k4[k4_i] -= 1
                                for k5_i in triple_to_k5s[t]:
                                    rc_k5[k5_i] -= 1
                                # Update violations
                                for k4_i in triple_to_k4s[t]:
                                    if rc_k4[k4_i] == 4:
                                        violations += 1
                                    elif rc_k4[k4_i] == 3:
                                        violations -= 1
                                for k5_i in triple_to_k5s[t]:
                                    if rc_k5[k5_i] == 0:
                                        violations += 1
                                    elif rc_k5[k5_i] == 1:
                                        violations -= 1
                                fixed = True
                                break
                    else:
                        # Try to fix a blue K5
                        violated_k5s = [i for i, r in enumerate(rc_k5) if r == 0]
                        if violated_k5s:
                            k5_idx = random.choice(violated_k5s)
                            t_indices = k5_triples[k5_idx]
                            # Flip one random triple in this K5 to red
                            t = random.choice(t_indices)
                            if colors[t] == 1:  # It's blue
                                # Flip to red
                                colors[t] = 0
                                for k4_i in triple_to_k4s[t]:
                                    rc_k4[k4_i] += 1
                                for k5_i in triple_to_k5s[t]:
                                    rc_k5[k5_i] += 1
                                # Update violations
                                for k4_i in triple_to_k4s[t]:
                                    if rc_k4[k4_i] == 4:
                                        violations += 1
                                    elif rc_k4[k4_i] == 3:
                                        violations -= 1
                                for k5_i in triple_to_k5s[t]:
                                    if rc_k5[k5_i] == 0:
                                        violations += 1
                                    elif rc_k5[k5_i] == 1:
                                        violations -= 1
                                fixed = True
                                break
                    if fixed:
                        break
                
                if not fixed:
                    break
            
            if violations == 0:
                return colors
            
            # Simulated annealing phase
            T = 1.0
            cooling = 0.9995
            min_T = 0.01
            
            current_time = time.time()
            steps = 0
            max_steps = 100000
            
            while steps < max_steps and time.time() < current_time + restart_time_limit:
                if time.time() > deadline - 5:
                    break
                    
                # Pick random triple
                t = random.randint(0, num_triples - 1)
                old_color = colors[t]
                new_color = 1 - old_color
                
                delta = 0
                
                # Update K4s
                for k4_idx in triple_to_k4s[t]:
                    if old_color == 0:  # Red -> Blue
                        if rc_k4[k4_idx] == 4:
                            delta -= 1
                    else:  # Blue -> Red
                        if rc_k4[k4_idx] == 3:
                            delta += 1
                
                # Update K5s
                for k5_idx in triple_to_k5s[t]:
                    if old_color == 0:  # Red -> Blue
                        if rc_k5[k5_idx] == 0:
                            delta -= 1
                    else:  # Blue -> Red
                        if rc_k5[k5_idx] == 0:
                            delta -= 1
                
                # Accept move
                if delta < 0 or (random.random() < pow(2.718, -delta / max(T, 0.001))):
                    colors[t] = new_color
                    for k4_idx in triple_to_k4s[t]:
                        if old_color == 0:
                            rc_k4[k4_idx] -= 1
                        else:
                            rc_k4[k4_idx] += 1
                    for k5_idx in triple_to_k5s[t]:
                        if old_color == 0:
                            rc_k5[k5_idx] -= 1
                        else:
                            rc_k5[k5_idx] += 1
                    violations += delta
                    if violations == 0:
                        return colors
                    T *= cooling
                    if T < min_T:
                        T = min_T
                else:
                    T *= cooling
                    if T < min_T:
                        T = min_T
                steps += 1
        
        return None

    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 15:
            break
        colors = solve_n(n)
        if colors is not None:
            return {'n': n, 'colors': colors}
            
    # Fallback: return a small valid coloring
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
