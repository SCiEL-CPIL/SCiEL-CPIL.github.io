import time
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute for each a in [1, 90], the set of (b, c) pairs with a < b < c <= 180
    # such that 1/a = 1/b + 1/c, i.e., (b-a)(c-a) = a^2
    # For each divisor d of a^2 where d < a, we get b = a + d, c = a + a^2//d
    # We need b > a and c > b, so d < a^2/d, i.e., d < a
    conflict_pairs = {}
    for a in range(1, 91):
        a2 = a * a
        pairs = set()
        for d in range(1, a):  # d must be < a for b > a and c > b
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and b <= 180 and c <= 180:
                    pairs.add((b, c))
        conflict_pairs[a] = pairs
    
    # Precompute reverse: for each (b,c), which a values conflict
    # Also precompute for each a, the list of (b,c) pairs as sorted list for faster lookup
    conflict_pairs_list = {a: sorted(pairs) for a, pairs in conflict_pairs.items()}
    
    # For faster checking: for each a, precompute the set of b values that could be in a conflict
    # b = a + d where d | a^2 and d < a
    a_to_b_values = {}
    for a in range(1, 91):
        a2 = a * a
        b_vals = set()
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and b <= 180 and c <= 180:
                    b_vals.add(b)
        a_to_b_values[a] = b_vals
    
    # Precompute for each pair (x, y) with x < y, what a values would make (a, x, y) a conflict
    # 1/a = 1/x + 1/y => a = 2xy/(x+y)
    # So a must be an integer, meaning (x+y) | 2xy
    pair_to_a = {}
    for x in range(1, 181):
        for y in range(x+1, 181):
            num = 2 * x * y
            den = x + y
            if num % den == 0:
                a = num // den
                if 1 <= a < x and a <= 90:  # a must be smallest
                    if a not in pair_to_a:
                        pair_to_a[a] = set()
                    pair_to_a[a].add((x, y))
    
    S = set(range(91, 181))
    
    def can_add_fast(a, S):
        """Check if adding 'a' to S keeps it valid."""
        if a > 90:
            # a is the largest possible element in any triple
            # Only need to check: is there x < y < a in S such that 1/x = 1/y + 1/a?
            # 1/x = 1/y + 1/a => y = ax/(a-x), need x < y < a
            # Or: is there x < a and y < a in S such that 1/x = 1/a + 1/y? Same thing.
            # Or: is there x < y in S with x < a, y < a such that 1/x = 1/y + 1/a?
            # Actually if a > 90, a can only be the largest c in a triple (a,b,c) with a<b<c
            # But c = a > 90, and b > a > 90, so b > 90. Then 1/a = 1/b + 1/c means
            # a < b and a < c, so a is smallest. But we said a > 90. Contradiction?
            # Wait: in triple (a,b,c) with a<b<c, we have 1/a = 1/b + 1/c.
            # If a > 90, then b > 90 and c > 90. So all three are > 90.
            # But our base set is [91,180], so if a,b,c are all in [91,180],
            # we need to check if any triple in [91,180] satisfies the equation.
            # For a in [91,180], b > a, c > b, all <= 180.
            # (b-a)(c-a) = a^2. Since b-a >= 1 and c-a >= 2, we need a^2 >= 2.
            # Also b = a + d, c = a + a^2/d where d | a^2 and 1 <= d < a.
            # b <= 180 means a + d <= 180, so d <= 180 - a <= 89.
            # c <= 180 means a + a^2/d <= 180, so a^2/d <= 180 - a.
            # For a >= 91: a^2/d <= 89, so d >= a^2/89 >= 91^2/89 = 8281/89 ≈ 93.
            # But d <= 89. Contradiction! So no triples exist with all elements > 90.
            # Therefore, adding any a > 90 is always safe.
            return True
        
        # a <= 90, a is the smallest element in any triple involving it
        # Check: for each b in S with b > a, if (b-a) | a^2, then c = a + a^2/(b-a)
        # must not be in S (and c > b)
        a2 = a * a
        for b in S:
            if b <= a:
                continue
            d = b - a
            if a2 % d != 0:
                continue
            c = a + a2 // d
            if c > b and c in S:
                return False
        
        # Case 2: a is the middle element (b) in triple (x, a, c) with x < a < c
        # 1/x = 1/a + 1/c => c = ax/(a-x), need x < a and c > a
        for x in S:
            if x >= a:
                continue
            denom = a - x
            num = a * x
            if num % denom != 0:
                continue
            c = num // denom
            if c > a and c in S:
                return False
        
        # Case 3: a is the largest element (c) in triple (x, y, a) with x < y < a
        # 1/x = 1/y + 1/a => y = ax/(a-x), need x < y < a
        for x in S:
            if x >= a:
                continue
            denom = a - x
            num = a * x
            if num % denom != 0:
                continue
            y = num // denom
            if x < y < a and y in S:
                return False
        
        return True
    
    # Greedily add smaller numbers from 1 to 90
    for a in range(1, 91):
        if time.time() > deadline - 5:
            break
        if can_add_fast(a, S):
            S.add(a)
    
    best_S = set(S)
    best_size = len(S)
    
    # Local search with hill climbing
    random.seed(123)
    all_nums = set(range(1, 181))
    
    excluded = list(all_nums - S)
    
    # Precompute conflict info for faster swap decisions
    # For each excluded element a, precompute which elements in S would block it
    # This is complex, so we'll use a simpler approach: try adding, if it fails,
    # identify the blocking triple and try removing one of its other elements
    
    def find_blocking_triple(a, S):
        """If a cannot be added, find a triple that blocks it."""
        if a > 90:
            return None
        a2 = a * a
        # Case 1: a is smallest
        for b in S:
            if b <= a:
                continue
            d = b - a
            if a2 % d != 0:
                continue
            c = a + a2 // d
            if c > b and c in S:
                return (a, b, c)
        # Case 2: a is middle
        for x in S:
            if x >= a:
                continue
            denom = a - x
            num = a * x
            if num % denom != 0:
                continue
            c = num // denom
            if c > a and c in S:
                return (x, a, c)
        # Case 3: a is largest
        for x in S:
            if x >= a:
                continue
            denom = a - x
            num = a * x
            if num % denom != 0:
                continue
            y = num // denom
            if x < y < a and y in S:
                return (x, y, a)
        return None
    
    # Improved local search: try to add, if blocked, try targeted swaps
    swap_count = 0
    for iteration in range(300000):
        if time.time() > deadline - 2:
            break
        
        if not excluded:
            break
            
        a = random.choice(excluded)
        
        if can_add_fast(a, S):
            S.add(a)
            excluded.remove(a)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
        else:
            # Find blocking triple and try to remove one of its other elements
            blocking = find_blocking_triple(a, S)
            if blocking:
                # Try removing each element in the blocking triple (except a)
                for elem in blocking:
                    if elem == a:
                        continue
                    if elem not in S:
                        continue
                    # Try: remove elem, add a
                    S.remove(elem)
                    excluded.append(elem)
                    
                    if can_add_fast(a, S):
                        S.add(a)
                        excluded.remove(a)
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                        swap_count += 1
                    else:
                        # Revert
                        S.add(elem)
                        excluded.remove(elem)
            else:
                # No specific blocking triple found (shouldn't happen), try random swap
                if S:
                    b = random.choice(list(S))
                    S.remove(b)
                    excluded.append(b)
                    
                    if can_add_fast(a, S):
                        S.add(a)
                        excluded.remove(a)
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                        swap_count += 1
                    else:
                        # Revert
                        S.add(b)
                        excluded.remove(b)
    
    # Second pass: try different orderings for greedy addition
    if time.time() < deadline - 10:
        # Try greedy with random ordering
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            S2 = set(range(91, 181))
            small_nums = list(range(1, 91))
            random.shuffle(small_nums)
            
            for a in small_nums:
                if time.time() > deadline - 3:
                    break
                if can_add_fast(a, S2):
                    S2.add(a)
            
            if len(S2) > best_size:
                best_S = set(S2)
                best_size = len(S2)
    
    return sorted(best_S)
