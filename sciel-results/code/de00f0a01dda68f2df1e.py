import time
import math

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 22
    candidates = list(range(1, N + 1))
    
    def time_ok():
        return time.time() - start_time < timeout

    def backtrack(current, available, target_size, lcm_val, sums_set):
        if not time_ok():
            return None
            
        if len(current) == target_size:
            return current
            
        if len(current) + len(available) < target_size:
            return None
            
        for i, x in enumerate(available):
            if not time_ok():
                return None
                
            new_lcm = math.lcm(lcm_val, x)
            
            # If LCM doesn't change, R=1. If it changes, R = new_lcm / lcm_val
            # Note: lcm_val is always a divisor of new_lcm
            R = new_lcm // lcm_val
            
            # The new value for x in the scaled system
            v_new = new_lcm // x
            
            # Check for collisions
            # The new sums will be { s * R + v_new * bit | s in sums_set, bit in {0,1} }
            # Actually, we just need to check if any new sum collides with an existing one (scaled) or another new one.
            # Since we are adding one element, the new sums are $S_{scaled} \cup (S_{scaled} + v_{new})$.
            # $S_{scaled} = \{ s \cdot R \mid s \in sums\_set \}$
            
            # We can check if $S_{scaled} \cup (S_{scaled} + v_{new})$ has size $2 \cdot |sums\_set|$.
            # This is equivalent to checking if $(S_{scaled} + v_{new})$ intersects $S_{scaled}$?
            # No, it's checking if the two sets are disjoint.
            # Intersection of $S_{scaled}$ and $S_{scaled} + v_{new}$ is non-empty if there exist $a, b \in sums\_set$ such that $a \cdot R = b \cdot R + v_{new}$.
            # i.e., $(a - b) \cdot R = v_{new}$.
            
            # However, we also need to ensure that within the new sums, there are no collisions among themselves?
            # The new set is $S_{new} = \{ s \cdot R \mid s \in S_{old} \} \cup \{ s \cdot R + v_{new} \mid s \in S_{old} \}$.
            # Collisions can happen:
            # 1. Within first part: impossible if S_old was distinct and R > 0.
            # 2. Within second part: impossible if S_old was distinct.
            # 3. Between first and second: $s_1 \cdot R = s_2 \cdot R + v_{new} \implies (s_1 - s_2) \cdot R = v_{new}$.
            
            # So we just need to check if there exists a difference $d \in \{ s_1 - s_2 \mid s_1, s_2 \in sums\_set \}$ such that $d \cdot R = v_{new}$.
            # Or simpler: construct the new set and check its length.
            # Since $|sums\_set| = 2^k$, and $k \le 15$ (trying to find 16), $2^{15} = 32768$.
            # Constructing the set is fast.
            
            # Optimization:
            # scaled_sums = {s * R for s in sums_set}
            # new_sums = scaled_sums | {s + v_new for s in scaled_sums}
            # if len(new_sums) != 2 * len(sums_set): collision
            
            # But we can do it in one pass or use set operations.
            
            # Let's try the set construction approach, it's clear and likely fast enough for k<16.
            # For k=15, 32k elements. Set ops are C-optimized.
            
            # However, we can prune earlier.
            # If v_new is small, collisions are more likely?
            
            # Let's just do the set update.
            
            # To avoid creating huge temporary sets if not needed, we can check intersection first?
            # Intersection of scaled_sums and (scaled_sums + v_new)
            # This is equivalent to checking if v_new is in { (s1 - s2) * R | s1, s2 in sums_set }
            
            # Let's stick to set union for simplicity and speed (C level).
            
            # One more check: if R > 1, the values grow.
            
            # Implementation:
            # scaled = {s * R for s in sums_set}
            # if not scaled: # shouldn't happen
            #     pass
            # shifted = {s + v_new for s in scaled}
            # if scaled & shifted: # Collision
            #     continue
            # new_sums = scaled | shifted
            
            # Wait, `scaled & shifted` checks if any element in `scaled` is in `shifted`.
            # This is exactly the condition for collision between the "exclude x" and "include x" subsets.
            
            # If no collision, new_sums is valid.
            
            # We need to pass `new_sums` to the next recursion.
            # But we don't want to build `new_sums` if we are just checking.
            # We only build it if we recurse.
            # But we need it for the next level.
            
            # So:
            # 1. Check collision.
            # 2. If no collision, build new_sums and recurse.
            
            # Collision check:
            # Is there s1, s2 in sums_set such that s1 * R == s2 * R + v_new?
            # => (s1 - s2) * R == v_new
            # => s1 - s2 == v_new / R
            # Since s1, s2 are integers, v_new must be divisible by R?
            # v_new = new_lcm / x. R = new_lcm / lcm_val.
            # v_new / R = (new_lcm / x) / (new_lcm / lcm_val) = lcm_val / x.
            # So we need to check if there is a difference $d \in \Delta(sums\_set)$ such that $d = \text{lcm\_val} / x$.
            # Note: $\text{lcm\_val} / x$ is an integer if $x$ divides $\text{lcm\_val}$.
            # Since $x$ is a new number added to `current`, and `lcm_val` is LCM of `current`, $x$ might not divide `lcm_val`.
            # Wait, $R = \text{new\_lcm} / \text{lcm\_val}$.
            # $v_{new} = \text{new\_lcm} / x$.
            # Condition: $s_1 R = s_2 R + v_{new} \iff (s_1 - s_2) R = v_{new} \iff s_1 - s_2 = v_{new} / R$.
            # $v_{new} / R = (\text{new\_lcm} / x) / (\text{new\_lcm} / \text{lcm\_val}) = \text{lcm\_val} / x$.
            # This requires $x$ to divide $\text{lcm\_val}$.
            # If $x$ does not divide $\text{lcm\_val}$, then $\text{lcm\_val} / x$ is not an integer, so no integer $s_1 - s_2$ can equal it.
            # Thus, if $x \nmid \text{lcm\_val}$, there is NO collision between the two halves!
            # This is a huge pruning opportunity.
            
            # Let's verify this logic.
            # Sums are integers.
            # $s_1, s_2$ are integers.
            # $s_1 R - s_2 R = v_{new}$
            # $R(s_1 - s_2) = v_{new}$
            # $s_1 - s_2 = v_{new} / R = \text{lcm\_val} / x$.
            # If $\text{lcm\_val} \% x != 0$, then RHS is not integer. LHS is integer. Contradiction.
            # So if $x$ does not divide current LCM, the new element is "dissociated" from the existing sums in the sense that no collision occurs between the subset that includes x and the one that doesn't, *relative to the scaling*.
            # Wait, does this mean we can always add x if $x \nmid \text{lcm\_val}$?
            # No, we must also ensure that the new sums themselves don't collide with each other?
            # The new sums are $S_{scaled}$ and $S_{scaled} + v_{new}$.
            # We proved they are disjoint.
            # Are elements within $S_{scaled}$ distinct? Yes, $S_{old}$ was distinct and $R > 0$.
            # Are elements within $S_{scaled} + v_{new}$ distinct? Yes.
            # So YES, if $x \nmid \text{lcm\_val}$, the new set is dissociated.
            
            # This is a massive speedup. We only need to perform the expensive set check if $x \mid \text{lcm\_val}$.
            
            if lcm_val % x == 0:
                # Need to check for collision
                # Target difference: d = lcm_val // x
                # Check if d is in the difference set of sums_set?
                # Or just check intersection as before, which is safer.
                # Intersection check:
                # scaled = {s * R for s in sums_set}
                # shifted = {s + v_new for s in scaled}
                # if scaled & shifted: continue
                
                # Optimization for intersection check:
                # We know we are looking for $s_1 R = s_2 R + v_{new}$
                # $s_1 R - s_2 R = v_{new}$
                # $R (s_1 - s_2) = v_{new}$
                # $s_1 - s_2 = v_{new} / R = \text{lcm\_val} / x$
                # Let $D = \text{lcm\_val} // x$.
                # We need to check if there exist $s_1, s_2 \in sums\_set$ such that $s_1 - s_2 = D$.
                # This is equivalent to checking if $(sums\_set + D) \cap sums\_set$ is not empty?
                # No, $s_1 = s_2 + D$. So $s_2 + D \in sums\_set$.
                # So we check if any $s_2 + D$ is in $sums\_set$.
                
                D = lcm_val // x
                # Check if (sums_set + D) intersects sums_set
                # We can iterate over sums_set and check if s + D is in sums_set.
                # This is O(2^k) lookups.
                
                # To make it even faster, we can precompute the difference set? No, sums_set changes.
                # But checking $s + D \in sums\_set$ is fast.
                
                has_collision = False
                for s in sums_set:
                    if s + D in sums_set:
                        has_collision = True
                        break
                        # Note: We don't need to check s - D?
                        # If s1 - s2 = D, then s1 = s2 + D.
                        # Iterating s2 and checking s2 + D covers all pairs.
                        
                if has_collision:
                    continue
                
                # If no collision, we need to build new_sums for the next step.
                # new_sums = {s * R for s in sums_set} | {s * R + v_new for s in sums_set}
                # R = new_lcm // lcm_val
                # v_new = new_lcm // x
                
                # We can build this efficiently.
                # Since we already know there is no collision, the size is 2 * len(sums_set).
                
                # However, we must update the LCM and Sums for the next recursion.
                # We can construct new_sums now.
                
                R_val = new_lcm // lcm_val
                v_new_val = new_lcm // x
                
                # To save time, we can create the new set.
                # new_sums = set()
                # for s in sums_set:
                #    new_sums.add(s * R_val)
                #    new_sums.add(s * R_val + v_new_val)
                
                # This is O(2^k).
                
                new_sums = set()
                for s in sums_set:
                    base = s * R_val
                    new_sums.add(base)
                    new_sums.add(base + v_new_val)
                    
                # Recurse
                result = backtrack(current + [x], available[i+1:], target_size, new_lcm, new_sums)
                if result is not None:
                    return result
                    
            else:
                # No collision possible.
                # Just update state and recurse.
                # R = new_lcm // lcm_val
                # v_new = new_lcm // x
                # new_sums = {s * R for s in sums_set} | {s * R + v_new for s in sums_set}
                
                R_val = new_lcm // lcm_val
                v_new_val = new_lcm // x
                
                new_sums = set()
                for s in sums_set:
                    base = s * R_val
                    new_sums.add(base)
                    new_sums.add(base + v_new_val)
                    
                result = backtrack(current + [x], available[i+1:], target_size, new_lcm, new_sums)
                if result is not None:
                    return result
                    
        return None

    # Try to find the largest possible set
    # Start from N down to 1
    for k in range(N, 0, -1):
        if not time_ok():
            break
            
        # Heuristic: Sort candidates?
        # Maybe larger numbers first? They are more likely to divide LCM?
        # Or smaller numbers?
        # Let's stick to ascending.
        
        available = sorted(candidates)
        
        # Initial state:
        # current = []
        # lcm_val = 1
        # sums_set = {0}
        
        result = backtrack([], available, k, 1, {0})
        
        if result is not None:
            return result
            
    # Fallback
    return [1, 2, 4, 8, 16]
