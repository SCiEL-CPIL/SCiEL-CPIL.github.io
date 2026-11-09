import time
import random

def construct():
    n = 192
    max_time = 55.0
    
    # Start with a greedy set from high to low
    S = []
    S_set = set()
    for x in range(n, 0, -1):
        if is_valid_add(x, S, S_set):
            S.append(x)
            S_set.add(x)
            
    best_S = list(S)
    best_size = len(S)
    
    start_time = time.time()
    
    # Simulated Annealing
    current_S = list(best_S)
    current_set = set(current_S)
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < max_time:
        # Random move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            candidates = [x for x in range(1, n+1) if x not in current_set]
            if not candidates: continue
            x = random.choice(candidates)
            new_S = current_S + [x]
            new_set = current_set | {x}
            
            if is_valid_add(x, current_S, current_set):
                # Accept
                current_S = new_S
                current_set = new_set
                if len(current_S) > best_size:
                    best_S = list(current_S)
                    best_size = len(current_S)
        elif move_type == 'remove':
            if not current_S: continue
            x = random.choice(current_S)
            new_S = [y for y in current_S if y != x]
            new_set = current_set - {x}
            # Removing is always valid
            current_S = new_S
            current_set = new_set
        elif move_type == 'swap':
            if not current_S: continue
            x = random.choice(current_S)
            candidates = [y for y in range(1, n+1) if y not in current_set]
            if not candidates: continue
            y = random.choice(candidates)
            
            # Check if y is valid in S \ {x}
            temp_S = [z for z in current_S if z != x]
            temp_set = current_set - {x}
            
            if is_valid_add(y, temp_S, temp_set):
                current_S = temp_S + [y]
                current_set = temp_set | {y}
                if len(current_S) > best_size:
                    best_S = list(current_S)
                    best_size = len(current_S)
        
        T *= cooling
        
    return best_S

def is_valid_add(x, S, S_set):
    # S is list, S_set is set
    # Check 1: x does not divide sum of any 3 in S
    # For each w in S, check if exists u,v in S\{w} such that (u+v) % x == (-w) % x
    
    if len(S) >= 3:
        for w in S:
            target = (-w) % x
            # Check pair sum mod x in S \ {w}
            if not exists_pair_sum_mod(S, w, target, x):
                # If no pair sum, continue
                pass
            else:
                # Found a pair u,v such that u+v+w is divisible by x
                return False
                
    # Check 2: For each a in S, a does not divide x + u + v for u,v in S\{a}
    for a in S:
        target = (-x) % a
        if exists_pair_sum_mod(S, a, target, a):
            return False
            
    return True

def exists_pair_sum_mod(S, exclude, target, mod):
    # Check if there exist u, v in S \ {exclude} such that (u + v) % mod == target
    # S is list, exclude is element to ignore
    # We can use a set of residues and counts
    
    # Optimization: If mod is large, we can just use a set of values?
    # No, we need residues.
    
    # Build residue counts for S \ {exclude}
    # Since N is small, we can just do a nested loop or single loop with a set.
    # Let's use a set of residues and a set of values for quick check.
    
    # Actually, just iterating is fast enough for N=50.
    # But we call this O(N^2) times.
    # Let's try to be smart.
    
    # We can precompute residues for S? No, S changes.
    
    # Simple approach:
    # 1. Create a set of residues in S \ {exclude}
    # 2. For each u in S \ {exclude}, check if (target - u) % mod is in the set.
    # 3. Handle the case where u is the only element with that residue.
    
    residues = set()
    for u in S:
        if u != exclude:
            residues.add(u % mod)
            
    for u in S:
        if u != exclude:
            needed_residue = (target - u) % mod
            if needed_residue in residues:
                # Check if there is a v != u with this residue
                # If u % mod != needed_residue, then any v with that residue is distinct from u
                if u % mod != needed_residue:
                    return True
                else:
                    # We need another element with the same residue
                    # Count occurrences
                    count = 0
                    for v in S:
                        if v != exclude and v % mod == needed_residue:
                            count += 1
                            if count > 1:
                                return True
                    # If count == 1, it's just u, so no pair
                    pass
    return False
