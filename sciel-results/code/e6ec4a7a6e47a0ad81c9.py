import time
import random
import numpy as np

def construct():
    N = 128
    deadline = time.time() + 55.0
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    M = len(aps)
    
    # Map position to list of (ap_index, role)
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for j, pos in enumerate(ap):
            pos_to_aps[pos].append((idx, j))

    def is_violation(c0, c1, c2, c3):
        # Violation if distinct colors < 3
        # Equivalent to set size 1 or 2
        # Fast path: if all 4 same -> violation
        if c0 == c1 and c1 == c2 and c2 == c3:
            return True
        # If 3 same -> violation
        if c0 == c1 and c1 == c2:
            return True
        if c0 == c1 and c1 == c3:
            return True
        if c0 == c2 and c2 == c3:
            return True
        if c1 == c2 and c2 == c3:
            return True
        # If 2 same, 2 same (pairs) -> violation
        # Patterns: A A B B, A B A B, A B B A
        if c0 == c1 and c2 == c3:
            return True
        if c0 == c2 and c1 == c3:
            return True
        if c0 == c3 and c1 == c2:
            return True
        
        return False

    def count_violations_fast(colors):
        total = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if is_violation(c0, c1, c2, c3):
                total += 1
        return total

    def greedy_construct(K, seed):
        """
        Greedy construction: assign colors 0..K-1 to positions 0..N-1.
        For each position, pick a color that minimizes new violations with previously assigned positions.
        """
        random.seed(seed)
        colors = [-1] * N
        # We'll track violations incrementally, but for simplicity in greedy,
        # we can just check against already assigned neighbors.
        
        # Actually, a smarter greedy: for each position, try each color 0..K-1,
        # and count how many APs involving this position and already-assigned positions
        # would be violated. Pick the color with min violations.
        
        # To make it faster and more effective, we can randomize the tie-breaking
        # or the order of colors.
        
        # Let's use a heuristic: for each position, evaluate all K colors.
        # Cost = number of APs containing this position where all other positions
        # are already assigned AND the AP would be a violation.
        
        for pos in range(N):
            best_color = -1
            best_cost = float('inf')
            best_ties = []
            
            # Shuffle colors for diversity
            color_order = list(range(K))
            random.shuffle(color_order)
            
            for c in color_order:
                cost = 0
                for ap_idx, role in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    # Check if all other positions in this AP are assigned
                    all_assigned = True
                    for j, p in enumerate(ap):
                        if p == pos:
                            continue
                        if colors[p] == -1:
                            all_assigned = False
                            break
                    if not all_assigned:
                        continue
                    
                    # Construct the tuple with candidate color
                    if role == 0:
                        nc = (c, colors[ap[1]], colors[ap[2]], colors[ap[3]])
                    elif role == 1:
                        nc = (colors[ap[0]], c, colors[ap[2]], colors[ap[3]])
                    elif role == 2:
                        nc = (colors[ap[0]], colors[ap[1]], c, colors[ap[3]])
                    else:
                        nc = (colors[ap[0]], colors[ap[1]], colors[ap[2]], c)
                    
                    if is_violation(*nc):
                        cost += 1
                
                if cost < best_cost:
                    best_cost = cost
                    best_color = c
                    best_ties = [c]
                elif cost == best_cost:
                    best_ties.append(c)
            
            # Pick randomly from ties
            colors[pos] = random.choice(best_ties)
        
        return colors

    def local_search(colors, K, max_steps=50000):
        """
        Local search to fix violations and reduce color count.
        Returns (colors, distinct_colors, total_violations)
        """
        # Initialize violation state
        ap_violation = [False] * M
        total_violations = 0
        for idx, ap in enumerate(aps):
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if is_violation(c0, c1, c2, c3):
                ap_violation[idx] = True
                total_violations += 1
        
        color_counts = [0] * K
        for c in colors:
            color_counts[c] += 1
        distinct_colors = sum(1 for c in color_counts if c > 0)
        
        T = 5.0
        cooling = 0.9995
        best_E = total_violations * 1000 + distinct_colors
        best_colors = colors[:]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick random position
            pos = random.randint(0, N - 1)
            old_color = colors[pos]
            new_color = random.randint(0, K - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate Delta E
            delta_v = 0
            for ap_idx, role in pos_to_aps[pos]:
                ap = aps[ap_idx]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                old_v = ap_violation[ap_idx]
                
                if role == 0:
                    nc = (new_color, c1, c2, c3)
                elif role == 1:
                    nc = (c0, new_color, c2, c3)
                elif role == 2:
                    nc = (c0, cc1, new_color, c3) # Typo fix below
                else:
                    nc = (c0, c1, c2, new_color)
                
                # Correct tuple construction
                if role == 0:
                    nc = (new_color, c1, c2, c3)
                elif role == 1:
                    nc = (c0, new_color, c2, c3)
                elif role == 2:
                    nc = (c0, c1, new_color, c3)
                else:
                    nc = (c0, c1, c2, new_color)
                
                new_v = is_violation(*nc)
                
                if new_v and not old_v:
                    delta_v += 1
                elif not new_v and old_v:
                    delta_v -= 1
            
            # Update distinct colors
            delta_d = 0
            if color_counts[old_color] == 1:
                delta_d -= 1
            if color_counts[new_color] == 0:
                delta_d += 1
            
            delta_E = delta_v * 1000 + delta_d
            
            # Accept?
            if delta_E < 0:
                accept = True
            else:
                if T > 1e-6:
                    accept = random.random() < np.exp(-delta_E / T)
                else:
                    accept = False
            
            if accept:
                colors[pos] = new_color
                total_violations += delta_v
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
                distinct_colors += delta_d
                
                for ap_idx, role in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    ap_violation[ap_idx] = is_violation(c0, c1, c2, c3)
                
                current_E = total_violations * 1000 + distinct_colors
                if current_E < best_E:
                    best_E = current_E
                    best_colors = colors[:]
            
            T *= cooling
        
        return best_colors, distinct_colors, best_E // 1000

    best_solution = None
    best_distinct = N

    # Try K from 3 upwards
    for K in range(3, 10):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple seeds for this K
        for seed in range(20):
            if time.time() > deadline:
                break
            
            # Greedy init
            colors = greedy_construct(K, seed)
            
            # Local search polish
            colors, distinct, viol = local_search(colors, K, max_steps=5000)
            
            if viol == 0:
                if distinct < best_distinct:
                    best_solution = colors
                    best_distinct = distinct
                found = True
                break
        
        if found:
            break
    
    if best_solution is None:
        # Fallback: All distinct
        best_solution = list(range(N))
        
    return best_solution
