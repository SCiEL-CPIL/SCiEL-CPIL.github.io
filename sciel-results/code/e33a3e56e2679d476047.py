import itertools
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    if k == 5 and g == 9:
        # Try Cayley graph on A6
        try:
            n, edges = find_cayley_a6()
            if n is not None:
                return n, edges
        except Exception:
            pass
        
        # Try Cayley graph on PSL(2, 8)
        try:
            n, edges = find_cayley_psl28()
            if n is not None:
                return n, edges
        except Exception:
            pass
            
    # Fallback to seed logic (L-U-W)
    # ... (copy seed code)
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

def find_cayley_a6():
    # A6 elements
    # We can use permutations of 0..5
    # Involutions are double transpositions
    import itertools
    
    elements = []
    # Generate all even permutations of 0..5
    # This is 360 elements
    # We can just iterate and filter
    # Or use a library? No, standard library.
    
    # Efficient way:
    # Start with identity
    # Generate all permutations, filter by parity
    
    # Actually, we can just use the 45 involutions directly?
    # No, we need the group to multiply them.
    
    # Let's generate the group A6
    # This might be slow if done naively, but 360 is small.
    
    # Better: Precompute the multiplication table for the 45 involutions?
    # No, product of two involutions is not necessarily an involution.
    
    # We need the full group.
    
    # Let's use a simple representation: tuple of 6 ints.
    
    # Generate all permutations of range(6)
    # Filter for even parity
    
    perms = []
    for p in itertools.permutations(range(6)):
        # Check parity
        # Count inversions
        inv = 0
        for i in range(6):
            for j in range(i+1, 6):
                if p[i] > p[j]:
                    inv += 1
        if inv % 2 == 0:
            perms.append(p)
            
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute multiplication table
    # mul[i][j] = index of perm_i * perm_j
    # Note: Permutation composition (p * q)(x) = p(q(x))
    # In Python, if we store tuples, we need to be careful with order.
    # Let's define multiplication as composition.
    
    n_elems = len(perms)
    mul_table = [[0]*n_elems for _ in range(n_elems)]
    for i in range(n_elems):
        pi = perms[i]
        for j in range(n_elems):
            pj = perms[j]
            # Compute pi(pj(x)) for all x
            prod = tuple(pi[pj[x]] for x in range(6))
            mul_table[i][j] = perm_to_idx[prod]
            
    # Identify involutions
    involutions = []
    for i in range(n_elems):
        if mul_table[i][i] == 0: # Identity is 0?
             # Check if perms[i] is identity
             if perms[i] == (0,1,2,3,4,5):
                 continue
             # Check if p*p = identity
             if mul_table[i][i] == 0:
                 involutions.append(i)
                 
    # Wait, identity is index 0?
    # Let's ensure identity is 0.
    # The first even permutation in itertools is (0,1,2,3,4,5)?
    # Yes, if we iterate in lexicographic order.
    
    # So involutions are indices i where i != 0 and p_i^2 = e.
    
    # Now search for 5 generators
    # We want no short relations.
    # Check if all words of length <= 4 are distinct.
    
    import random
    random.seed(42)
    
    # Shuffle involutions to find a good set faster?
    # Or just iterate combinations.
    # 1.2 million combinations.
    # Each check: 781 multiplications.
    # 1.2M * 781 ~ 1e9 operations. Might be too slow in Python.
    
    # Optimization:
    # We can prune early.
    # Check length 1: 5 elements.
    # Check length 2: 25 elements.
    # Check length 3: 125 elements.
    # Check length 4: 625 elements.
    
    # We can do this incrementally.
    
    # But 1e9 is too high.
    # We need a faster check or fewer candidates.
    
    # Maybe we can just pick random sets and check them?
    # Probability of success?
    # If girth 9 is rare, random might take long.
    
    # Let's try to limit the search time.
    # If we don't find one in 10 seconds, give up.
    
    start_search = time.time()
    
    # Use a heuristic: pick generators that are "far apart" in some sense?
    # Or just random sampling.
    
    # Let's try random sampling of 5 involutions.
    # 100,000 samples.
    
    for _ in range(100000):
        if time.time() - start_search > 20:
            break
        S = random.sample(involutions, 5)
        # Check girth
        # Generate words of length 0 to 4
        # Use a set to store reached elements
        reached = set()
        reached.add(0) # Identity
        
        # Length 1
        current_level = [0]
        valid = True
        for length in range(1, 5):
            next_level = []
            for v in current_level:
                for s in S:
                    w = mul_table[v][s]
                    if w in reached:
                        # Check if it's a back edge?
                        # If w is in reached, and it's not the parent?
                        # In a tree, if we hit a visited node, we have a cycle.
                        # But we are just checking if the map is injective.
                        # If w is already in reached, it means two different words map to the same element.
                        # This implies a relation.
                        # So valid = False
                        valid = False
                        break
                    reached.add(w)
                    next_level.append(w)
            if not valid:
                break
            current_level = next_level
            
        if valid:
            # Found a valid set
            # Construct the graph
            # Vertices are 0..359
            # Edges: (i, mul_table[i][s]) for each s in S
            # But we only need the connected component?
            # Cayley graph is connected if S generates the group.
            # Do 5 random involutions generate A6?
            # A6 is simple, generated by any two elements of coprime order?
            # Involutions have order 2.
            # Two involutions generate a dihedral group or A5 or A6?
            # It is likely they generate A6.
            # We can check connectivity.
            
            # Check connectivity
            visited = {0}
            queue = [0]
            while queue:
                v = queue.pop()
                for s in S:
                    w = mul_table[v][s]
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
                        
            if len(visited) == n_elems:
                # Connected
                edges = []
                for v in range(n_elems):
                    for s in S:
                        w = mul_table[v][s]
                        if v < w:
                            edges.append((v, w))
                return n_elems, edges
    return None, None

def find_cayley_psl28():
    # Similar logic for PSL(2, 8)
    # Order 504
    # This is more complex to implement from scratch.
    # Skip for now, rely on A6.
    return None, None
