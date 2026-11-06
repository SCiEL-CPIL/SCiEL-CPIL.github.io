import time
import sys
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 128
    # Precompute all 4-term progressions
    # A 4-term progression is (a, a+d, a+2d, a+3d) with d > 0 and a+3d <= N
    # Using 0-based indices for positions 0..127
    aps = []
    for a in range(N):
        for d in range(1, N // 3 + 1):
            end = a + 3 * d
            if end < N:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map index to list of AP indices it belongs to
    ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices[pos].append(idx)
            
    # Degree of each position
    degree = [len(lst) for lst in ap_indices]
    
    # Order positions by degree (most constrained first)
    # We will assign in this order, but keep track of original indices
    order = sorted(range(N), key=lambda i: -degree[i])
    
    # For forward checking, we need to quickly check if an AP is still satisfiable.
    # An AP (a,b,c,d) with colors ca,cb,cc,cd (-1 for unassigned).
    # Let S be the set of assigned colors.
    # Let U be the number of unassigned positions.
    # We need |S ∪ {new_colors}| >= 3.
    # Let r = 3 - |S|.
    # If r <= 0: OK.
    # If r > U: Fail.
    # If r == 1:
    #   If U == 1: The unassigned position must not have a color in S.
    #   If U > 1: At least one unassigned position must not have a color in S.
    #   (This is hard to check locally without propagation, but we can check if ALL unassigned 
    #    are forced to be in S? No, we don't know that yet. 
    #    However, if we assign a color, we can check if it makes the AP unsat.)
    #    Simple check: if r > 0, we need to ensure that it is possible to pick r distinct 
    #    new colors from the unassigned positions.
    #    If r == 1 and U == 1, the single unassigned position CANNOT be in S.
    #    If r == 2 and U == 2, the two unassigned positions MUST be distinct and NOT in S.
    
    def is_satisfiable(ap_idx, colors):
        a, b, c, d = aps[ap_idx]
        vals = [colors[a], colors[b], colors[c], colors[d]]
        assigned = [v for v in vals if v != -1]
        unassigned_count = len(vals) - len(assigned)
        
        if not assigned:
            return True
            
        S = set(assigned)
        r = 3 - len(S)
        
        if r <= 0:
            return True
        if r > unassigned_count:
            return False
            
        # If r == 1 and unassigned_count == 1:
        # The single unassigned variable must take a color NOT in S.
        # This is a constraint on the domain of that variable.
        # For simple pruning, we just return True (it's possible).
        # But we can check if the variable's domain is empty?
        # We don't maintain domains explicitly here, just colors array.
        # So we just check if it's *possible*.
        
        # If r == 2 and unassigned_count == 2:
        # The two variables must take distinct colors not in S.
        
        return True

    def check_constraint(pos, color, colors):
        """
        Check if assigning `color` to `pos` violates any constraint involving `pos`.
        Returns True if valid, False otherwise.
        """
        for ap_idx in ap_indices[pos]:
            # Temporarily set
            a, b, c, d = aps[ap_idx]
            # We need to check if this assignment makes the AP unsatisfiable.
            # We can just set it and call is_satisfiable, then unset.
            # Or better, logic directly.
            
            # Get current values
            vals = [colors[a], colors[b], colors[c], colors[d]]
            # Update the value at pos
            # Find index of pos in (a,b,c,d)
            # This is slow. Let's just set/unset.
            
            old_val = colors[pos]
            colors[pos] = color
            if not is_satisfiable(ap_idx, colors):
                colors[pos] = old_val
                return False
            colors[pos] = old_val
            
        return True

    def solve_k(k, colors, assigned_count, order_idx):
        """
        Backtracking with forward checking.
        k: number of colors
        colors: list of N colors, -1 for unassigned
        assigned_count: number of positions assigned
        order_idx: index in the `order` list of next position to assign
        """
        if time.time() > deadline:
            return None
            
        if assigned_count == N:
            return colors[:]
            
        pos = order[order_idx]
        
        # Symmetry breaking:
        # The first position assigned (in our order) must be 0.
        # Subsequent positions can use 0..min(k-1, number of distinct colors used so far).
        # Actually, standard symmetry breaking:
        # If we are at the first position in the order, it must be 0.
        # If not, it can be any color in 0..k-1, but we can restrict to 
        # 0..(max_color_used_so_far + 1) to avoid permuting color names.
        
        if assigned_count == 0:
            domain = [0]
        else:
            # Find max color used so far
            max_used = -1
            for p in range(N):
                if colors[p] != -1 and colors[p] > max_used:
                    max_used = colors[p]
            
            # We can use colors 0..min(k-1, max_used + 1)
            # Wait, if we haven't used color 2 yet, can we use it now?
            # Yes, if k > max_used.
            # So domain is 0..min(k-1, max_used + 1)
            max_allowed = min(k - 1, max_used + 1)
            domain = list(range(max_allowed + 1))
            
            # Shuffle for randomness? Or keep deterministic?
            # Randomness helps escape local minima if we add restarts.
            # For now, let's try deterministic first to see if it finds a solution quickly.
            # Actually, the directive says "randomized greedy ... explores ... more effectively".
            # Let's shuffle the domain.
            random.shuffle(domain)

        for color in domain:
            if not check_constraint(pos, color, colors):
                continue
                
            colors[pos] = color
            result = solve_k(k, colors, assigned_count + 1, order_idx + 1)
            if result is not None:
                return result
            colors[pos] = -1
            
        return None

    # Try k from 3 upwards
    for k in range(3, 13):
        # Multiple restarts for each k to handle randomness
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            colors = [-1] * N
            # Fix one variable to break symmetry? 
            # The solver handles symmetry breaking via domain restriction.
            # But we can also fix the first variable in the order to 0 to speed up.
            # The solver does this automatically (assigned_count == 0 -> domain [0]).
            
            result = solve_k(k, colors, 0, 0)
            if result is not None:
                return result

    # Fallback: Greedy
    # Use a simple greedy algorithm that picks the smallest valid color.
    # Order by degree.
    colors = [-1] * N
    for pos in order:
        for color in range(128):
            valid = True
            for ap_idx in ap_indices[pos]:
                # Check if assigning `color` to `pos` keeps AP satisfiable
                # We need to check if the AP would have < 3 colors if we fill the rest?
                # No, we just need to ensure that the current assignment doesn't 
                # make it IMPOSSIBLE to have 3 colors.
                
                a, b, c, d = aps[ap_idx]
                vals = [colors[a], colors[b], colors[c], colors[d]]
                # Set pos to color
                # Find index
                # This is inefficient but fallback is rare.
                
                # Let's just check the final constraint:
                # If we assign color to pos, does the AP become unsatisfiable?
                # We can use a simple check:
                # Count distinct colors in assigned positions of this AP.
                # If count == 2 and all other positions are unassigned, we need at least 1 new color.
                # If count == 1 and 2 positions unassigned, we need 2 new colors.
                
                # Simpler: Just assign and check if any AP has < 3 colors 
                # considering that unassigned can be anything?
                # No, that's not right.
                
                # Let's just use the `is_satisfiable` logic.
                old_val = colors[pos]
                colors[pos] = color
                if not is_satisfiable(ap_idx, colors):
                    valid = False
                    colors[pos] = old_val
                    break
                colors[pos] = old_val
                
            if valid:
                colors[pos] = color
                break
        else:
            # Should not happen
            colors[pos] = 0

    return colors
# EVOLVE-BLOCK-END
