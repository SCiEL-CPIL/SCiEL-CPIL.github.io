import time
import random

def construct():
    n = 10
    deadline = time.time() + 55  # safety margin before the 90s kill

    # ---- Deterministic construction of 9 MOLS via Z5 x Z2 ----
    # Elements of Z5 x Z2 are encoded as a*2 + b where a in Z5, b in Z2
    # We use 9 different linear forms to generate 9 orthogonal Latin squares
    
    def encode(a, b):
        return a * 2 + b

    def decode(x):
        return x // 2, x % 2

    # The 9 slopes for the Latin squares
    # Each square L_k[i][j] is defined by a linear combination
    # We need 9 mutually orthogonal Latin squares
    
    # For Z5 x Z2, we can construct MOLS using the fact that 
    # Z5 x Z2 is a group of order 10, and we can use linear maps
    
    # A standard construction: for each pair (a, b) where a in Z5\{0}, b in Z2
    # plus some additional ones, we get a Latin square
    
    # Actually, let's use a known construction for order 10
    # The maximum number of MOLS of order 10 is 9 (since 10 = 2*5, and 
    # for prime powers p^k, we can get p^k - 1 MOLS)
    
    # But 10 is not a prime power, so we need a different approach
    # However, it is known that 9 MOLS of order 10 exist
    
    # Let's use the construction based on Z5 x Z2
    # For each (r, s) where r in Z5, s in Z2, not both zero, we can define
    # L_{(r,s)}[i][j] = (r*i_5 + s*i_2, r*j_5 + s*j_2) ... wait, this needs more care
    
    # Actually, let me use a simpler known construction:
    # Since 10 = 2 * 5, we can use the fact that there exist 9 MOLS of order 10
    # One way is to use the field GF(25) and identify it with Z5 x Z2 in a compatible way
    
    # For simplicity, let's just return 2 well-known orthogonal Latin squares
    # which we know work, and try to find more via search if time permits
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]

    L2 = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        ai, bi = decode(i)
        for j in range(n):
            aj, bj = decode(j)
            a_part = (ai + 2 * aj) % 5
            b_part = (bi + bj) % 2
            L2[i][j] = encode(a_part, b_part)

    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
        for j in range(n):
            if sorted([sq[i][j] for i in range(n)]) != list(range(n)):
                return False
        return True

    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    if not (is_latin(L1) and is_latin(L2) and are_orthogonal(L1, L2)):
        # Should not happen; fall back to a single cyclic square.
        return [[[int((i + j) % n) for j in range(n)] for i in range(n)]]

    base = [L1, L2]
    base_int = [[[int(L1[i][j]) for j in range(n)] for i in range(n)],
                [[int(L2[i][j]) for j in range(n)] for i in range(n)]]

    # ---- Try to find more squares orthogonal to all existing ones, via backtracking ----
    
    def search_next(base_sq, rng):
        """Find a new Latin square orthogonal to all squares in base_sq"""
        L = [[-1] * n for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_sets = [set() for _ in base_sq]  # pair_sets[k] stores (base_sq[k][i][j], val)

        def valid(i, j, val):
            if val in row_used[i]:
                return False
            if val in col_used[j]:
                return False
            for k, sq in enumerate(base_sq):
                if (sq[i][j], val) in pair_sets[k]:
                    return False
            return True

        cells = [(i, j) for i in range(n) for j in range(n)]

        def backtrack(remaining):
            if time.time() > deadline:
                return False
            if not remaining:
                return True

            # MRV: pick the cell with fewest valid values.
            best_k = -1
            best_vals = None
            best_cnt = 11
            for k, (i, j) in enumerate(remaining):
                vals = [v for v in range(n) if valid(i, j, v)]
                c = len(vals)
                if c == 0:
                    return False
                if c < best_cnt:
                    best_cnt = c
                    best_k = k
                    best_vals = vals
                    if c == 1:
                        break

            i, j = remaining[best_k]
            new_remaining = remaining[:best_k] + remaining[best_k + 1:]
            rng.shuffle(best_vals)
            for val in best_vals:
                L[i][j] = val
                row_used[i].add(val)
                col_used[j].add(val)
                for k in range(len(base_sq)):
                    pair_sets[k].add((base_sq[k][i][j], val))
                if backtrack(new_remaining):
                    return True
                row_used[i].remove(val)
                col_used[j].remove(val)
                for k in range(len(base_sq)):
                    pair_sets[k].remove((base_sq[k][i][j], val))
                L[i][j] = -1
            return False

        if backtrack(cells):
            return L
        return None

    # Try to find additional squares
    # We want to maximize the number, so keep trying until we can't find more or time runs out
    max_squares = 9
    while len(base) < max_squares and time.time() < deadline:
        found = None
        # Try a few random restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
            rng = random.Random(12345 + len(base) * 1000 + attempt * 7)
            res = search_next(base, rng)
            if res is not None:
                # Verify orthogonality with all existing squares
                valid_sq = True
                if not is_latin(res):
                    valid_sq = False
                else:
                    for sq in base:
                        if not are_orthogonal(sq, res):
                            valid_sq = False
                            break
                if valid_sq:
                    found = res
                    break
        
        if found is not None:
            base.append(found)
            base_int.append([[int(found[i][j]) for j in range(n)] for i in range(n)])
        else:
            # Can't find another square with this base, break
            break

    # Return between 1 and 9 squares, all Python int, all lists.
    return base_int
