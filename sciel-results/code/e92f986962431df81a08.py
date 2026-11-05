import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    selected = []
    # Try greedy descending
    # We can also try other orders if time permits, but let's stick to one strong heuristic first.
    # Actually, a common trick for these problems is to pick numbers from the top half.
    # Let's just use the optimized greedy descending.
    
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if x can be added to selected
        # 1. x must not divide sum of any 3 distinct elements in selected
        valid = True
        if len(selected) >= 3:
            for t in itertools.combinations(selected, 3):
                if sum(t) % x == 0:
                    valid = False
                    break
        
        # 2. For each a in selected, a must not divide sum of any 3 distinct elements in (selected + [x]) \ {a}
        # The triples involving x are {x, u, v} where u, v in selected, u != v.
        # The sum is x + u + v.
        # We need to check if any a in selected divides x + u + v.
        # Note: u and v must be distinct from a.
        if valid:
            for a in selected:
                # We need to check if a divides x + u + v for any u, v in selected \ {a}, u != v
                # This is equivalent to checking all pairs (u, v) from selected where u != v and u != a and v != a
                # But we can iterate u, v in combinations of selected, 2
                # If u == a or v == a, skip? No, the triple is {x, u, v}. The "other" elements are x, u, v.
                # The divisor is a. a is NOT in {x, u, v} because a is in selected and x is new, and u,v are from selected.
                # Wait, if a is in selected, and u, v are in selected, it is possible that u == a?
                # The triple of summands is {x, u, v}. The divisor is a.
                # The condition is: a does not divide sum({x, u, v}).
                # The summands must be distinct OTHER selected integers.
                # So the set of summands is {x, u, v}.
                # a is a selected integer.
                # If a is in {x, u, v}, then a is not "OTHER".
                # x is not a (x is new, a is old).
                # So we must ensure u != a and v != a.
                for u, v in itertools.combinations(selected, 2):
                    if u == a or v == a:
                        continue
                    s = x + u + v
                    if s % a == 0:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            selected.append(x)
            
    return selected
