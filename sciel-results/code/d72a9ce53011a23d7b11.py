"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # Greedy: add x if it doesn't form a 3-AP with any two already selected
    def is_valid(selected, x):
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                a, b = selected[i], selected[j]
                # Check if x, a, b form a 3-AP in any order
                # 3-AP: a, b, c with b-a = c-b, i.e., 2b = a+c
                # So for triple (x, a, b): check if 2*min = max + mid... 
                # Actually, just check all permutations: 2*b == a + x, 2*a == b + x, 2*x == a + b
                if 2 * a == b + x or 2 * b == a + x or 2 * x == a + b:
                    return False
        return True
    
    selected = []
    for x in range(1, n+1):
        if is_valid(selected, x):
            selected.append(x)
    
    # Local search: try to improve by swapping
    def reciprocal_sum(s):
        return sum(1.0 / x for x in s)
    
    def has_3ap(s):
        sset = set(s)
        for a in s:
            for b in s:
                if a < b:
                    c = 2*b - a
                    if c > b and c in sset:
                        return True
                    c2 = 2*a - b
                    if c2 > 0 and c2 in sset and c2 != a and c2 != b:
                        # Need to check if c2, a, b form AP: 2*a = c2 + b? No, 2*a = c2 + b means c2 = 2a-b
                        # The AP is c2, a, b if 2*a = c2 + b, which is true by construction
                        # But we need c2 != a and c2 != b, and c2 in s
                        if c2 != a and c2 != b and c2 in sset:
                            return True
        return False
    
    # Better has_3ap check
    def has_3ap(s):
        sset = set(s)
        for a in s:
            for b in s:
                if a < b:
                    # a, b, 2b-a
                    c = 2*b - a
                    if c > b and c in sset:
                        return True
        return False
    
    current_sum = reciprocal_sum(selected)
    improved = True
    while improved:
        improved = False
        sset = set(selected)
        # Try removing one and adding another
        for remove_idx in range(len(selected)):
            for add_x in range(1, n+1):
                if add_x in sset:
                    continue
                new_set = set(selected)
                new_set.remove(selected[remove_idx])
                new_set.add(add_x)
                if not has_3ap(list(new_set)):
                    new_sum = reciprocal_sum(new_set)
                    if new_sum > current_sum + 1e-12:
                        selected = list(new_set)
                        current_sum = new_sum
                        improved = True
                        break
            if improved:
                break
    
    return selected
# EVOLVE-BLOCK-END
