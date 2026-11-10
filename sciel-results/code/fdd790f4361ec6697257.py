import math

def construct():
    H = set(range(129, 257))
    L = list(range(2, 129))
    
    # Precompute conflicts in H for each a in L
    C = {}
    for a in L:
        C[a] = set()
        for b in H:
            if math.lcm(a, b) <= 256:
                C[a].add(b)
                
    # Precompute conflicts within L
    conflict = {}
    for a in L:
        conflict[a] = set()
        for b in L:
            if a != b and math.lcm(a, b) <= 256:
                conflict[a].add(b)
                
    # Sort L by descending weight (1/a)
    L_sorted = sorted(L, key=lambda x: 1.0/x, reverse=True)
    
    best_score = 0.0
    best_SL = []
    
    # Greedy pass to initialize best_score
    current_SL = []
    current_U = set()
    current_score = 0.0
    
    for a in L_sorted:
        if any(b in conflict[a] for b in current_SL):
            continue
        new_conflicts = C[a] - current_U
        gain = 1.0/a - sum(1.0/b for b in new_conflicts)
        if gain > 0:
            current_SL.append(a)
            current_U.update(C[a])
            current_score += gain
            
    best_score = current_score
    best_SL = list(current_SL)
    
    # Precompute inverse of H elements for speed
    inv_H = {b: 1.0/b for b in H}
    
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(index, current_SL, current_U, current_score):
        nonlocal best_score, best_SL
        
        if index == len(L_sorted):
            if current_score > best_score:
                best_score = current_score
                best_SL = list(current_SL)
            return
            
        # Calculate upper bound
        # Sum of 1/a for all remaining compatible a
        remaining_potential = 0.0
        for i in range(index, len(L_sorted)):
            a = L_sorted[i]
            # Check if a is compatible with current_SL
            compat = True
            for b in current_SL:
                if a in conflict[b]:
                    compat = False
                    break
            if compat:
                remaining_potential += 1.0/a
                
        if current_score + remaining_potential <= best_score:
            return
            
        a = L_sorted[index]
        
        # Check if a is compatible with current_SL
        compat = True
        for b in current_SL:
            if a in conflict[b]:
                compat = False
                break
                
        # Option 1: Include a
        if compat:
            new_SL = current_SL + [a]
            new_U = current_U | C[a]
            # Calculate marginal gain
            # Penalty is sum of 1/b for b in C[a] not in current_U
            penalty = 0.0
            for b in C[a]:
                if b not in current_U:
                    penalty += inv_H[b]
            new_score = current_score + 1.0/a - penalty
            dfs(index + 1, new_SL, new_U, new_score)
            
        # Option 2: Exclude a
        dfs(index + 1, current_SL, current_U, current_score)
        
    dfs(0, [], set(), 0.0)
    
    # Construct final result
    final_SL = best_SL
    removed_from_H = set()
    for a in final_SL:
        removed_from_H.update(C[a])
        
    final_SH = [b for b in H if b not in removed_from_H]
    
    return final_SL + final_SH
