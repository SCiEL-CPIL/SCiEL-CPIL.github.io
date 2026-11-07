"""
Program for the suite task: construct a large set in 1..128 with no triple a,b,c
such that a+b, a+c, b+c are all also in the set.

Approach: Greedy construction.
Iterate through numbers 1 to 128. For each number x, check if adding it to the
current set S would create a forbidden triple.
A forbidden triple is a set of three distinct elements {a, b, c} from S such that
a+b, a+c, and b+c are all in S.

When adding x, a new forbidden triple can only form if:
1. x is one of the members {a, b, c}. Let the triple be {x, u, v}.
   Then we need x+u in S, x+v in S, and u+v in S.
2. x is one of the sums. Let the triple be {a, b, c} subset of S.
   Assume a+b = x. Then we need a+c in S and b+c in S.

If adding x does not create such a triple, we add it.
"""

def construct():
    S = set()
    
    for x in range(1, 129):
        can_add = True
        
        # Case 1: x is a member of the triple {x, u, v}
        # We need distinct u, v in S such that:
        # (x + u) in S, (x + v) in S, and (u + v) in S.
        
        # Find all u in S such that x+u is also in S.
        candidates = []
        for u in S:
            if (x + u) in S:
                candidates.append(u)
        
        # Check if any two distinct candidates u, v have u+v in S.
        found_bad_case1 = False
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                u = candidates[i]
                v = candidates[j]
                # u and v are distinct because they come from a set and indices are different
                if (u + v) in S:
                    found_bad_case1 = True
                    break
            if found_bad_case1:
                break
        
        if found_bad_case1:
            can_add = False
        else:
            # Case 2: x is one of the sums of a triple {a, b, c} subset of S.
            # Assume a + b = x.
            # We need a + c in S and b + c in S for some c in S distinct from a and b.
            
            found_bad_case2 = False
            # Iterate through all pairs (a, b) in S such that a + b = x
            for a in S:
                b = x - a
                if b in S and a != b and a < b:
                    # Check if there exists c in S, c != a, c != b
                    # such that a+c in S and b+c in S
                    for c in S:
                        if c == a or c == b:
                            continue
                        if (a + c) in S and (b + c) in S:
                            found_bad_case2 = True
                            break
                    if found_bad_case2:
                        break
            
            if found_bad_case2:
                can_add = False
        
        if can_add:
            S.add(x)
            
    return [int(x) for x in sorted(S)]
