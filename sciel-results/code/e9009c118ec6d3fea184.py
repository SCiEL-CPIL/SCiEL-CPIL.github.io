import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110  # Safety margin
    
    def covers_all(code, n, q, radius):
        # Check if code covers {0..q-1}^n
        # For n=4, 81 words. Fast.
        # For n=5, 243 words. Fast.
        # For n=8, 6561 words. Slower but ok for few checks.
        covered = [False] * (q ** n)
        for c in code:
            # Generate all neighbors within radius
            # For n=4, radius 1: 1 + 4*3 = 13 neighbors.
            # For n=5, radius 1: 1 + 5*3 = 16 neighbors.
            # Use itertools.product for neighbors?
            # Better: iterate over all words in space and check dist?
            # Space is small.
            for w in itertools.product(range(q), repeat=n):
                w_list = list(w)
                dist = sum(1 for i in range(n) if w_list[i] != c[i])
                if dist <= radius:
                    # Map w to index
                    idx = 0
                    for i, val in enumerate(w_list):
                        idx = idx * q + val
                    covered[idx] = True
        return all(covered)

    def get_neighbors(c, n, q, radius):
        # Yield all words within radius of c
        # For radius 1, just change 0 or 1 position.
        yield tuple(c)
        for i in range(n):
            for val in range(q):
                if val == c[i]: continue
                new_c = c[:]
                new_c[i] = val
                yield tuple(new_c)

    def find_min_code(n, q, radius, time_limit):
        # Greedy initialization
        space = list(itertools.product(range(q), repeat=n))
        uncovered = set(space)
        code = []
        
        while uncovered:
            # Find word in code with max coverage of uncovered?
            # Or find word in space with max uncovered neighbors?
            # Greedy: pick word from space that covers most uncovered.
            best_word = None
            max_cov = -1
            # Sample or iterate?
            # For n=4, 81 words. Iterate all.
            for w in space:
                if w in code: continue
                cov = 0
                for nb in get_neighbors(w, n, q, radius):
                    if nb in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_word = w
            code.append(best_word)
            for nb in get_neighbors(best_word, n, q, radius):
                uncovered.discard(nb)
                
        # Local Search: 1-swap
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(code)):
                # Try removing code[i]
                new_code = code[:i] + code[i+1:]
                if covers_all(new_code, n, q, radius):
                    code = new_code
                    improved = True
                    break
            if improved:
                continue
                
            # Try 1-swap: replace code[i] with some other word
            # Only if it allows removal later?
            # Just try random swaps or best swap?
            # For now, just removal is powerful.
            
        return code

    # 1. Try to find small code for n=4
    # If size 6, product with A^4 gives 486.
    # If size 7, product gives 567.
    
    best_code_8 = None
    best_size = float('inf')
    
    # Strategy 1: Product A^4 x C_4
    if time.time() < deadline:
        c4 = find_min_code(4, q, radius, time.time() + 20)
        size_4 = len(c4)
        # Construct product
        # C = [ (tail, head) for tail in A^4 for head in C4 ]
        # Or (head, tail)
        prod_code_4 = []
        tails = list(itertools.product(range(q), repeat=4))
        for t in tails:
            for h in c4:
                prod_code_4.append(list(t) + list(h))
        if covers_all(prod_code_4, 8, q, radius):
            if len(prod_code_4) < best_size:
                best_size = len(prod_code_4)
                best_code_8 = prod_code_4

    # Strategy 2: Product A^3 x C_5
    if time.time() < deadline:
        c5 = find_min_code(5, q, radius, time.time() + 30)
        size_5 = len(c5)
        prod_code_5 = []
        tails = list(itertools.product(range(q), repeat=3))
        for t in tails:
            for h in c5:
                prod_code_5.append(list(t) + list(h))
        if covers_all(prod_code_5, 8, q, radius):
            if len(prod_code_5) < best_size:
                best_size = len(prod_code_5)
                best_code_8 = prod_code_5

    # Strategy 3: Direct Greedy on n=8 (if time permits)
    # This is slow, so only if previous failed or time is left.
    if time.time() < deadline - 20:
        # Simple greedy on n=8
        space = list(itertools.product(range(q), repeat=8))
        uncovered = set(space)
        code = []
        while uncovered and time.time() < deadline - 10:
            best_word = None
            max_cov = -1
            # Sample 1000 random words from uncovered?
            # Or just pick first?
            # Greedy:
            for w in uncovered:
                cov = 0
                for nb in get_neighbors(w, 8, q, radius):
                    if nb in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_word = w
                # Break early if cov is max possible?
                if cov == 1 + 8*(q-1): break
            if best_word is None: break
            code.append(best_word)
            for nb in get_neighbors(best_word, 8, q, radius):
                uncovered.discard(nb)
                
        if covers_all(code, 8, q, radius):
            if len(code) < best_size:
                best_size = len(code)
                best_code_8 = code

    # Fallback to seed if nothing found or invalid
    if best_code_8 is None or not covers_all(best_code_8, 8, q, radius):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code_8 = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return best_code_8
