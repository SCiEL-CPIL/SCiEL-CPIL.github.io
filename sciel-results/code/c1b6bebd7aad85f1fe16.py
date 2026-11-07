"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 55
    n = 128
    
    # Precompute all forbidden configurations
    # A configuration is a set of values V such that if all v in V are in S, 
    # then there exist distinct a,b,c in S with a+b, a+c, b+c in S.
    # Specifically, V = {a, b, c, a+b, a+c, b+c} for distinct a,b,c.
    
    configs = []
    configs_by_val = [[] for _ in range(n)]
    
    # Iterate all distinct a, b, c
    # To avoid duplicates in configs (same set of values from different triples?),
    # we can just use a set of frozensets or bitmasks.
    # However, different triples might generate the same V?
    # Example: {1, 2, 3} -> sums 3, 4, 5. V={1,2,3,4,5}.
    # Is there another triple generating {1,2,3,4,5}?
    # If a,b,c are distinct, the set {a,b,c} is unique to the triple.
    # So V uniquely identifies the triple {a,b,c} if we assume a,b,c are the smallest 3?
    # Not necessarily.
    # But we just need to count conflicts.
    # If V is in S, it's a conflict.
    # We can just store unique bitmasks.
    
    seen_configs = set()
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 > n or s2 > n or s3 > n:
                    continue
                
                # Create bitmask for V
                # Values are 1-indexed in problem, 0-indexed in bit
                mask = 0
                vals = [a, b, c, s1, s2, s3]
                for v in vals:
                    mask |= (1 << (v - 1))
                
                if mask not in seen_configs:
                    seen_configs.add(mask)
                    configs.append(mask)
                    # Add to configs_by_val for each v in vals
                    for v in vals:
                        configs_by_val[v - 1].append(mask)
    
    # Local Search
    # Start with upper half
    best_mask = 0
    for i in range(64, n):
        best_mask |= (1 << i)
        
    # Calculate initial conflict count
    def count_conflicts(mask):
        cnt = 0
        for cfg in configs:
            if (mask & cfg) == cfg:
                cnt += 1
        return cnt
        
    current_mask = best_mask
    current_conf = count_conflicts(current_mask)
    current_pop = bin(current_mask).count('1')
    
    K = 10000
    current_score = K * current_pop - current_conf
    best_score = current_score
    best_valid_mask = current_mask if current_conf == 0 else 0
    
    # If initial is valid, keep it
    if current_conf == 0:
        best_valid_mask = current_mask
        best_size = current_pop
    else:
        best_valid_mask = 0
        best_size = 0
        
    T = 100.0
    min_T = 0.1
    
    while time.time() < deadline:
        # Pick a random bit to flip
        i = random.randint(0, n - 1)
        
        # Calculate delta
        bit = 1 << i
        
        if current_mask & bit:
            # Removing i
            # Conflicts removed: configs containing i that are currently satisfied
            removed_conf = 0
            for cfg in configs_by_val[i]:
                if (current_mask & cfg) == cfg:
                    removed_conf += 1
            
            delta_pop = -1
            delta_conf = -removed_conf
            delta_score = K * delta_pop - delta_conf
            
            if delta_score > 0 or (delta_score == 0 and random.random() < 0.5) or (delta_score < 0 and random.random() < pow(2.718, delta_score / T)):
                current_mask ^= bit
                current_pop += delta_pop
                current_conf += delta_conf
                current_score += delta_score
        else:
            # Adding i
            # Conflicts added: configs containing i where all other bits are set
            added_conf = 0
            for cfg in configs_by_val[i]:
                # Check if all bits in cfg except i are in current_mask
                # (current_mask & (cfg ^ bit)) == (cfg ^ bit)
                if (current_mask & (cfg ^ bit)) == (cfg ^ bit):
                    added_conf += 1
            
            delta_pop = 1
            delta_conf = added_conf
            delta_score = K * delta_pop - delta_conf
            
            if delta_score > 0 or (delta_score == 0 and random.random() < 0.5) or (delta_score < 0 and random.random() < pow(2.718, delta_score / T)):
                current_mask ^= bit
                current_pop += delta_pop
                current_conf += delta_conf
                current_score += delta_score
        
        # Update best
        if current_conf == 0:
            if current_pop > best_size:
                best_size = current_pop
                best_valid_mask = current_mask
        
        # Cool down
        T = max(min_T, T * 0.99)
        
    # Convert best_valid_mask to list
    result = []
    for i in range(n):
        if best_valid_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)

# EVOLVE-BLOCK-END
