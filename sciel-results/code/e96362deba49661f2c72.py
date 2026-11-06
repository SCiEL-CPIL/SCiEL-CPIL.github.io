"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    # Check if adding x to set S creates a forbidden triple
    # 1/a = 1/b + 1/c for distinct a,b,c in S ∪ {x}
    # Equivalent: (b-a)(c-a) = a² for some a, with b,c > a
    # Or equivalently: a = bc/(b+c) where b,c > a, and b+c divides bc
    
    def has_triple_with_x(S, x):
        """Check if adding x to S creates a triple 1/a=1/b+1/c with distinct elements."""
        S_set = set(S)
        S_set.add(x)
        elements = list(S_set)
        n = len(elements)
        # For each potential a (smallest in triple), check if there exist b,c > a with (b-a)(c-a) = a²
        # a must be the smallest of the three
        for a in elements:
            if a == x and len(S) == 0:
                continue
            # Check all pairs (b,c) from S_set where b > a, c > a, b != c
            # (b-a)(c-a) = a²
            a2 = a * a
            for b in elements:
                if b <= a:
                    continue
                ba = b - a
                if ba == 0:
                    continue
                if a2 % ba != 0:
                    continue
                ca = a2 // ba
                c = a + ca
                if c <= a:
                    continue
                if c == b:
                    continue
                if c in S_set:
                    return True
        return False
    
    # Greedy: try adding elements in different orders, keep best
    # Order 1: descending (large elements first - they're less likely to be 'a' in a triple)
    best_set = set()
    
    # Try multiple orderings and keep the best
    orderings = [
        list(range(180, 0, -1)),  # descending
        list(range(1, 181)),       # ascending
    ]
    
    # Also try: skip small numbers that are highly composite
    # Numbers with many divisors of a² are more constraining
    
    for order in orderings:
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            # Check if adding x creates a triple
            if not has_triple_with_x(S, x):
                S.add(x)
        if len(S) > len(best_set):
            best_set = S
    
    if not best_set:
        best_set = {180}
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
