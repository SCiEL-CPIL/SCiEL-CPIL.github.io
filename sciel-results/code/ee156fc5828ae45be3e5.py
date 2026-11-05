import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 110
    
    # Precompute weights
    weights = {a: 1.0 / (512 - a) for a in range(1, 512)}
    
    # Greedy initialization
    # Pick numbers with highest weight that are coprime to current set
    # Sort candidates by weight descending
    candidates = sorted(range(1, 512), key=lambda a: weights[a], reverse=True)
    
    chosen = []
    chosen_set = set()
    
    for a in candidates:
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            chosen_set.add(a)
    
    best_score = sum(weights[a] for a in chosen)
    best_set = list(chosen)
    
    def get_conflicts(v, chosen_set):
        """Return list of elements in chosen_set that are not coprime with v."""
        return [u for u in chosen_set if gcd(u, v) != 1]
    
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        
        # 1-Add Phase
        best_gain = 0
        best_v = None
        best_conflicts = None
        
        for v in range(511, 0, -1):
            if v in chosen_set:
                continue
            
            conflicts = get_conflicts(v, chosen_set)
            # Check if v is coprime to all non-conflicts
            # This is guaranteed by definition of conflicts
            
            gain = weights[v] - sum(weights[u] for u in conflicts)
            
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_v = v
                best_conflicts = conflicts
        
        if best_gain > 1e-12:
            # Apply move
            for u in best_conflicts:
                chosen_set.remove(u)
                chosen.remove(u)
            chosen_set.add(best_v)
            chosen.append(best_v)
            best_score += best_gain
            improved = True
            continue # Restart loop to check for new opportunities
        
        # 2-Add Phase
        # Try adding two numbers v1, v2
        # Gain = w(v1) + w(v2) - sum(w(u) for u in C(v1) U C(v2))
        
        # Optimization: Only consider v1, v2 that are not in chosen_set
        # And ideally, v1, v2 should be coprime to each other?
        # If gcd(v1, v2) != 1, they cannot both be in the set.
        # So we must check gcd(v1, v2) == 1.
        
        best_gain_2 = 0
        best_v1 = None
        best_v2 = None
        best_conflicts_union = None
        
        # Iterate over pairs
        # To save time, we can skip pairs where v1, v2 are not coprime
        # Also, we can restrict to high weight candidates
        
        # Let's iterate v1 from high weight to low
        # And v2 from high weight to low
        # Break early if potential max gain is low?
        
        for i in range(len(candidates)):
            if time.time() - start_time > DEADLINE:
                break
            v1 = candidates[i]
            if v1 in chosen_set:
                continue
            
            # Potential max gain for this v1 is w(v1) + max_w - 0
            # If w(v1) + max_w <= best_gain_2, we can't improve with this v1?
            # Not exactly, because conflicts reduce the gain.
            # But if w(v1) is very small, it's unlikely to help.
            
            for j in range(i+1, len(candidates)):
                v2 = candidates[j]
                if v2 in chosen_set:
                    continue
                
                # Check coprime
                if gcd(v1, v2) != 1:
                    continue
                
                # Calculate conflicts
                # C1 = conflicts(v1), C2 = conflicts(v2)
                # Union = C1 U C2
                
                # To avoid recalculating, we can do this:
                # But N is small, so let's just do it.
                
                c1 = get_conflicts(v1, chosen_set)
                c2 = get_conflicts(v2, chosen_set)
                
                # Union
                union_set = set(c1) | set(c2)
                union_list = list(union_set)
                
                gain = weights[v1] + weights[v2] - sum(weights[u] for u in union_list)
                
                if gain > best_gain_2 + 1e-12:
                    best_gain_2 = gain
                    best_v1 = v1
                    best_v2 = v2
                    best_conflicts_union = union_list
        
        if best_gain_2 > 1e-12:
            # Apply move
            for u in best_conflicts_union:
                chosen_set.remove(u)
                chosen.remove(u)
            chosen_set.add(best_v1)
            chosen_set.add(best_v2)
            chosen.append(best_v1)
            chosen.append(best_v2)
            best_score += best_gain_2
            improved = True
            continue # Restart
        
    return best_set
