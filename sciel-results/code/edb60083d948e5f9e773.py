import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_moduli = list(range(20, 201))
    
    def get_conflict_count(m, r, assign, other_moduli):
        """Calculate number of conflicts for modulus m with residue r against other_moduli."""
        count = 0
        for m2 in other_moduli:
            if m2 == m:
                continue
            r2 = assign[m2]
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                count += 1
        return count

    def local_search_fit(new_m, current_moduli, current_assign, max_iters=1000):
        """
        Tries to fit new_m into current_moduli by reassigning residues.
        Returns new assignment if conflicts == 0, else None.
        """
        target_moduli = current_moduli + [new_m]
        
        assign = dict(current_assign)
        assign[new_m] = random.randint(0, new_m - 1)
        
        # Calculate initial total conflicts
        # Sum of conflicts for each m
        # To be efficient, we can just sum conflicts for all pairs?
        # Or sum conflicts for each m (counting each pair twice? No, we can just sum for all m, it's symmetric).
        # Actually, just sum get_conflict_count for all m in target_moduli.
        # This is O(N^2).
        
        total_conf = 0
        for m in target_moduli:
            total_conf += get_conflict_count(m, assign[m], assign, target_moduli)
        total_conf //= 2 # Each conflict counted twice
        
        if total_conf == 0:
            return assign
        
        t = 1.0
        cooling = 0.95
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                return None
            
            m = random.choice(target_moduli)
            old_r = assign[m]
            new_r = random.randint(0, m - 1)
            if new_r == old_r:
                continue
            
            # Calculate delta in conflicts
            # Conflicts involving m change.
            # Old conflicts of m: old_conf_m
            # New conflicts of m: new_conf_m
            # Delta = new_conf_m - old_conf_m
            
            old_conf_m = get_conflict_count(m, old_r, assign, target_moduli)
            assign[m] = new_r
            new_conf_m = get_conflict_count(m, new_r, assign, target_moduli)
            
            delta = new_conf_m - old_conf_m
            
            if delta < 0 or random.random() < math.exp(-delta / t):
                total_conf += delta
                # Keep new_r
            else:
                assign[m] = old_r
                # Revert
            
            if total_conf == 0:
                return assign
            
            t *= cooling
        
        return None

    current_moduli = []
    current_assignment = {}
    rejected = []
    
    for m in sorted(all_moduli):
        # 1. Direct fit
        direct_fit = None
        for r in range(m):
            ok = True
            for m2, r2 in current_assignment.items():
                if r % math.gcd(m, m2) == r2 % math.gcd(m, m2):
                    ok = False
                    break
            if ok:
                direct_fit = r
                break
        
        if direct_fit is not None:
            current_assignment[m] = direct_fit
            current_moduli.append(m)
        else:
            # 2. Local search fit
            new_assign = local_search_fit(m, current_moduli, current_assignment, max_iters=200)
            if new_assign:
                current_assignment = new_assign
                current_moduli.append(m)
            else:
                rejected.append(m)
        
        if time.time() > deadline - 30.0:
            break

    # 3. Try to add rejected
    rejected.sort()
    for m in rejected:
        if time.time() > deadline - 10.0:
            break
        new_assign = local_search_fit(m, current_moduli, current_assignment, max_iters=200)
        if new_assign:
            current_assignment = new_assign
            current_moduli.append(m)
            
    return [[m, current_assignment[m]] for m in current_moduli]
