"""Rewritten program with stronger algorithm for 4-AP coloring."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    # Precompute all nonconstant 4-term progressions
    progressions = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            progressions.append((a, a + d, a + 2 * d, a + 3 * d))
    
    def count_violations(colors):
        """Count number of 4-APs with fewer than 3 distinct colors."""
        violations = 0
        for prog in progressions:
            a, b, c, d_pos = prog
            distinct = len({colors[a-1], colors[b-1], colors[c-1], colors[d_pos-1]})
            if distinct < 3:
                violations += 1
        return violations
    
    def check_valid(colors):
        """Check if a complete coloring is valid."""
        for prog in progressions:
            a, b, c, d_pos = prog
            if len({colors[a-1], colors[b-1], colors[c-1], colors[d_pos-1]}) < 3:
                return False
        return True
    
    def stochastic_local_search(k, max_iters, seed_val):
        """Stochastic local search: start random, flip colors to reduce violations."""
        rng = random.Random(seed_val)
        colors = [rng.randint(0, k - 1) for _ in range(n)]
        
        best_colors = colors[:]
        best_violations = count_violations(colors)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            violations = count_violations(colors)
            if violations == 0:
                return colors
            
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
            
            # Pick a random violated progression and try to fix it
            violated_progs = []
            for prog in progressions:
                a, b, c, d_pos = prog
                distinct = len({colors[a-1], colors[b-1], colors[c-1], colors[d_pos-1]})
                if distinct < 3:
                    violated_progs.append(prog)
            
            if not violated_progs:
                return colors
            
            prog = rng.choice(violated_progs)
            a, b, c, d_pos = prog
            positions = [a-1, b-1, c-1, d_pos-1]
            
            # Try flipping one of the positions to a random color
            pos_idx = rng.choice(positions)
            new_color = rng.randint(0, k - 1)
            
            old_color = colors[pos_idx]
            colors[pos_idx] = new_color
            
            new_violations = count_violations(colors)
            
            # Accept if better or equal (with some randomness for exploration)
            if new_violations <= violations or rng.random() < 0.1:
                # Keep the change
                pass
            else:
                # Revert
                colors[pos_idx] = old_color
        
        return best_colors if count_violations(best_colors) < count_violations(colors) else colors
    
    def greedy_construct(k, seed_val):
        """Greedy construction with lookahead."""
        rng = random.Random(seed_val)
        colors = [-1] * n
        
        # Precompute progressions by position for quick lookup
        pos_to_progs = [[] for _ in range(n + 1)]
        for idx, prog in enumerate(progressions):
            a, b, c, d_pos = prog
            pos_to_progs[a].append(idx)
            pos_to_progs[b].append(idx)
            pos_to_progs[c].append(idx)
            pos_to_progs[d_pos].append(idx)
        
        for pos in range(1, n + 1):
            if time.time() > deadline - 5:
                return None
            
            # Find valid colors for this position
            valid_colors = list(range(k))
            rng.shuffle(valid_colors)
            
            for color in valid_colors:
                colors[pos - 1] = color
                
                # Check if this causes any immediate violation
                consistent = True
                for prog_idx in pos_to_progs[pos]:
                    a, b, c, d_pos = progressions[prog_idx]
                    if (colors[a-1] != -1 and colors[b-1] != -1 and 
                        colors[c-1] != -1 and colors[d_pos-1] != -1):
                        if len({colors[a-1], colors[b-1], colors[c-1], colors[d_pos-1]}) < 3:
                            consistent = False
                            break
                
                if consistent:
                    break
                else:
                    colors[pos - 1] = -1
                    valid_colors.remove(color)
            
            if colors[pos - 1] == -1:
                return None
        
        if check_valid(colors):
            return colors
        return None
    
    best_result = None
    best_color_count = n + 1
    
    # Try increasing number of colors
    for k in range(3, 9):
        if time.time() > deadline - 15:
            break
        
        # Try greedy first (fast)
        for seed_val in range(30):
            if time.time() > deadline - 15:
                break
            result = greedy_construct(k, seed_val)
            if result is not None:
                used = len(set(result))
                if used < best_color_count:
                    best_result = result
                    best_color_count = used
                break
        
        # Try stochastic local search
        if time.time() < deadline - 20:
            for seed_val in range(50):
                if time.time() > deadline - 15:
                    break
                result = stochastic_local_search(k, 5000, seed_val)
                if result is not None and check_valid(result):
                    used = len(set(result))
                    if used < best_color_count:
                        best_result = result
                        best_color_count = used
                    break
        
        if best_result is not None and best_color_count <= k:
            break
    
    if best_result is None:
        # Fallback: use unique colors (guaranteed valid)
        best_result = list(range(n))
        best_color_count = n
    
    # Ensure colors are in range 0..127
    color_set = set(best_result)
    if max(color_set) > 127 or min(color_set) < 0:
        mapping = {c: i for i, c in enumerate(sorted(color_set))}
        best_result = [mapping[c] for c in best_result]
    
    return best_result
