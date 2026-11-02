import time
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 100s limit for search
    
    # Moore Bound calculation
    # For odd girth g, Moore bound M is 1 + k * sum_{i=0}^{(g-1)/2 - 1} (k-1)^i
    # Actually for girth g, radius is floor((g-1)/2).
    # M = 1 + k * sum_{j=0}^{floor((g-1)/2)-1} (k-1)^j
    radius = (g - 1) // 2
    moore = 1
    term = 1
    for _ in range(radius):
        moore += k * term
        term *= (k - 1)
    # For k=5, g=9, radius=4.
    # M = 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426.
    # So n >= 428 (even) or 427?
    # Circulant graphs with odd k require even n (for the self-inverse element).
    
    n_start = 428
    if n_start % 2 != 0:
        n_start += 1
        
    # Search for Circulant Graph C_n({a, -a, b, -b, n/2})
    # We want girth >= g.
    # This means no cycle of length < g.
    # We check for existence of cycles of length g-1 (if g-1 is even) or g-2?
    # If no cycle of length g-1, girth is at least g (if g-1 is even, we check g-1. If g is odd, g-1 is even).
    # For g=9, we check for cycles of length 8.
    
    target_cycle_len = g - 1  # 8
    
    def has_short_cycle(n, a, b, c, target_len):
        # c = n // 2
        # Generators: a, -a, b, -b, c
        # We look for a sequence of target_len generators that sums to 0 mod n,
        # with no backtracking (s_{i+1} != -s_i) and no repeated vertices (partial sums distinct).
        
        # Since target_len is small (8), we can use a set of partial sums.
        # But we need to check ALL sequences? No, we can use DP or just iterate?
        # Number of sequences is 5^8 = 390k.
        # For each (a,b), checking 390k sequences is too slow if we do it for many (a,b).
        
        # Optimization:
        # A cycle of length 8 in a circulant graph corresponds to a relation
        # x1 + x2 + ... + x8 = 0 mod n, xi in S.
        # We can pre-compute the set of all possible sums of 8 elements from S?
        # No, S depends on a, b.
        
        # Alternative:
        # Iterate over all 5^8 sequences of indices in {0,1,2,3,4} corresponding to S.
        # For a given sequence of signs/choices, the sum is a linear combination of a, b, c.
        # Sum = A*a + B*b + C*c = 0 mod n.
        # A, B, C are integers determined by the sequence.
        # We need to check if there exists a sequence such that:
        # 1. No backtracking.
        # 2. Partial sums are distinct (no 0 until end).
        # 3. A*a + B*b + C*c == 0 mod n.
        
        # There are 5^8 = 390625 sequences.
        # We can pre-compute the (A, B, C) coefficients and validity flags for each sequence ONCE.
        
        pass

    # Pre-compute sequence properties
    seqs = []
    generators_idx = [0, 1, 2, 3, 4] # 0:a, 1:-a, 2:b, 3:-b, 4:c
    # Map index to (coeff_a, coeff_b, coeff_c)
    # 0: (1, 0, 0)
    # 1: (-1, 0, 0)
    # 2: (0, 1, 0)
    # 3: (0, -1, 0)
    # 4: (0, 0, 1)
    
    for seq in itertools.product([0, 1, 2, 3, 4], repeat=target_cycle_len):
        # Check backtracking
        valid = True
        for i in range(target_cycle_len - 1):
            # Backtrack if seq[i] and seq[i+1] are inverses
            # 0 (a) and 1 (-a) are inverses
            # 2 (b) and 3 (-b) are inverses
            # 4 (c) and 4 (c) are inverses (c = -c)
            if (seq[i] == 0 and seq[i+1] == 1) or (seq[i] == 1 and seq[i+1] == 0):
                valid = False; break
            if (seq[i] == 2 and seq[i+1] == 3) or (seq[i] == 3 and seq[i+1] == 2):
                valid = False; break
            if seq[i] == 4 and seq[i+1] == 4:
                valid = False; break
        if not valid: continue
        
        # Check partial sums (no 0 before end)
        # This depends on a, b, c. So we can't pre-compute this part easily.
        # However, if the total sum is 0, and the sequence is "simple", it's likely a cycle.
        # We will check partial sums in the main loop.
        
        A = 0; B = 0; C = 0
        for s in seq:
            if s == 0: A += 1
            elif s == 1: A -= 1
            elif s == 2: B += 1
            elif s == 3: B -= 1
            elif s == 4: C += 1
            
        seqs.append((A, B, C, seq))
        
    # Filter sequences that sum to 0 for ANY a,b,c? No.
    # We need A*a + B*b + C*c == 0 mod n.
    
    # To speed up, we can group sequences by (A, B, C).
    # If (A, B, C) is (0,0,0), it's always 0. But backtracking/sum checks apply.
    # If (A, B, C) is non-zero, it's a specific linear equation.
    
    from collections import defaultdict
    seq_map = defaultdict(list)
    for A, B, C, seq in seqs:
        seq_map[(A, B, C)].append(seq)
        
    # We only care about (A,B,C) that are not all 0.
    # And we need to check if there exists a,b such that A*a + B*b + C*(n/2) == 0 mod n.
    
    # Iterate n
    for n in range(n_start, 1500, 2):
        if time.time() > deadline:
            break
        c = n // 2
        # Try random pairs (a, b) or structured?
        # Let's try a few random pairs for each n.
        # Or iterate a from 1 to n//2, and b from a+1 to n//2? Too slow.
        
        # Heuristic:
        # Pick a, b such that a, b, a+b, etc. are "large".
        # Just try a fixed number of random pairs.
        import random
        random.seed(42)
        found = False
        for _ in range(200): # 200 random trials per n
            a = random.randint(1, n-1)
            b = random.randint(1, n-1)
            if a == b or a == -a % n or b == -b % n or a == b % n: continue
            if a == c or b == c: continue # c is fixed generator
            
            # Check if any sequence gives 0
            is_valid = True
            for (A, B, C), seqs_list in seq_map.items():
                if A == 0 and B == 0 and C == 0:
                    continue
                val = (A * a + B * b + C * c) % n
                if val == 0:
                    # Found a zero sum. Now check partial sums for this specific seq.
                    # This is expensive.
                    # But if val != 0 for most, we skip.
                    for seq in seqs_list:
                        # Check partial sums
                        curr = 0
                        seen = {0}
                        valid_seq = True
                        for s in seq:
                            if s == 0: curr = (curr + a) % n
                            elif s == 1: curr = (curr - a) % n
                            elif s == 2: curr = (curr + b) % n
                            elif s == 3: curr = (curr - b) % n
                            elif s == 4: curr = (curr + c) % n
                            
                            if curr == 0:
                                if s != seq[-1]: # Not the last step
                                    valid_seq = False
                                    break
                            if curr in seen:
                                valid_seq = False
                                break
                            seen.add(curr)
                        
                        if valid_seq:
                            is_valid = False
                            break
                if not is_valid:
                    break
            
            if is_valid:
                # Found a valid graph!
                # Construct edges
                edges = []
                S = [a, -a % n, b, -b % n, c]
                for i in range(n):
                    for s in S:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
                found = True
                break
        if found:
            break

    # Fallback to LUW
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
