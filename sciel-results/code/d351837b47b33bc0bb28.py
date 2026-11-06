import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill
    N = 180

    # Precompute all forbidden triples: distinct a,b,c with 1/a = 1/b + 1/c
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    triples.append((a, b, c))
    triples = list(set(triples))
    num_triples = len(triples)

    # For each element, list of (other1, other2) pairs from triples containing it
    elem_triples = [[] for _ in range(N + 1)]
    for (a, b, c) in triples:
        elem_triples[a].append((b, c))
        elem_triples[b].append((a, c))
        elem_triples[c].append((a, b))

    # Element degree (number of triples containing it)
    elem_degree = [0] * (N + 1)
    for (a, b, c) in triples:
        elem_degree[a] += 1
        elem_degree[b] += 1
        elem_degree[c] += 1

    def can_add(z, S_mask):
        """Check if element z can be added to S_mask without creating a triple."""
        for (u, v) in elem_triples[z]:
            if (S_mask >> u) & 1 and (S_mask >> v) & 1:
                return False
        return True

    def count_violations(S_mask):
        """Count forbidden triples fully contained in S_mask."""
        count = 0
        for (a, b, c) in triples:
            if (S_mask >> a) & 1 and (S_mask >> b) & 1 and (S_mask >> c) & 1:
                count += 1
        return count

    def is_valid(S_mask):
        return count_violations(S_mask) == 0

    # --- Simulated Annealing over the hypergraph constraint ---
    # State: S_mask (bitmask of selected elements)
    # Energy: -|S| + penalty * num_violations(S)
    # Moves: add element, remove element, swap (remove one, add one)
    
    def simulated_annealing(init_mask, max_iters=50000):
        """Simulated annealing to maximize |S| subject to no triples."""
        T0 = 10.0
        T_end = 0.01
        cooling = (T_end / T0) ** (1.0 / max_iters) if max_iters > 0 else 1.0
        
        S = init_mask
        best_S = S
        best_size = bin(S).count('1')
        T = T0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            current_size = bin(S).count('1')
            current_viol = count_violations(S)
            current_energy = -current_size + 100.0 * current_viol
            
            # Choose a random move
            move_type = random.random()
            new_S = S
            
            if move_type < 0.3:
                # Add a random element not in S
                not_in_S = [z for z in range(1, N + 1) if not (S >> z) & 1]
                if not_in_S:
                    z = random.choice(not_in_S)
                    new_S = S | (1 << z)
            elif move_type < 0.5:
                # Remove a random element from S
                in_S = [z for z in range(1, N + 1) if (S >> z) & 1]
                if in_S:
                    z = random.choice(in_S)
                    new_S = S & ~(1 << z)
            else:
                # Swap: remove one, add one
                in_S = [z for z in range(1, N + 1) if (S >> z) & 1]
                not_in_S = [z for z in range(1, N + 1) if not (S >> z) & 1]
                if in_S and not_in_S:
                    x = random.choice(in_S)
                    y = random.choice(not_in_S)
                    new_S = (S & ~(1 << x)) | (1 << y)
            
            if new_S == S:
                continue
            
            new_size = bin(new_S).count('1')
            new_viol = count_violations(new_S)
            new_energy = -new_size + 100.0 * new_viol
            
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                S = new_S
                if new_size > best_size and new_viol == 0:
                    best_S = S
                    best_size = new_size
            
            T *= cooling
        
        return best_S, best_size

    # --- Conflict-driven repair: iteratively resolve violations ---
    def conflict_repair(S_mask, max_iters=10000):
        """Repair S_mask by removing elements from violated triples,
        then try to add back elements greedily."""
        S = S_mask
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Find all violated triples
            violated = []
            for (a, b, c) in triples:
                if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                    violated.append((a, b, c))
            
            if not violated:
                break
            
            # For each violated triple, pick the element to remove
            # Strategy: remove the element that appears in the most violated triples
            elem_viol_count = {}
            for (a, b, c) in violated:
                for e in (a, b, c):
                    elem_viol_count[e] = elem_viol_count.get(e, 0) + 1
            
            # Remove the most involved element
            if elem_viol_count:
                worst_elem = max(elem_viol_count, key=elem_viol_count.get)
                S &= ~(1 << worst_elem)
        
        # Now try to add back elements greedily
        for z in range(1, N + 1):
            if not (S >> z) & 1:
                if can_add(z, S):
                    S |= (1 << z)
        
        return S

    # --- Multi-start with diverse initializations ---
    best_mask = 0
    best_size = 0

    # Initializations to try
    inits = []
    
    # 1. Greedy removal from full set
    S = (1 << (N + 1)) - 2
    while True:
        violated_elems = [0] * (N + 1)
        has_violation = False
        for (a, b, c) in triples:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                has_violation = True
                violated_elems[a] += 1
                violated_elems[b] += 1
                violated_elems[c] += 1
        if not has_violation:
            break
        max_v = 0
        max_elem = -1
        for e in range(1, N + 1):
            if (S >> e) & 1 and violated_elems[e] > max_v:
                max_v = violated_elems[e]
                max_elem = e
        if max_elem == -1:
            break
        S &= ~(1 << max_elem)
    inits.append(S)
    
    # 2. Greedy add from empty, high degree first
    order_desc = sorted(range(1, N + 1), key=lambda x: -elem_degree[x])
    S = 0
    for z in order_desc:
        if can_add(z, S):
            S |= (1 << z)
    inits.append(S)
    
    # 3. Greedy add from empty, low degree first
    order_asc = sorted(range(1, N + 1), key=lambda x: elem_degree[x])
    S = 0
    for z in order_asc:
        if can_add(z, S):
            S |= (1 << z)
    inits.append(S)
    
    # 4. High-range seed
    S = 0
    for z in range(91, N + 1):
        S |= (1 << z)
    for z in range(1, 91):
        if can_add(z, S):
            S |= (1 << z)
    inits.append(S)
    
    # 5. Random initializations
    for _ in range(10):
        S = 0
        for z in range(1, N + 1):
            if random.random() < 0.6:
                S |= (1 << z)
        inits.append(S)
    
    # Process each initialization
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        # Conflict repair
        repaired = conflict_repair(init)
        rep_size = bin(repaired).count('1')
        if rep_size > best_size and is_valid(repaired):
            best_mask = repaired
            best_size = rep_size
        
        # Simulated annealing from repaired state
        if time.time() > deadline - 10:
            break
        sa_mask, sa_size = simulated_annealing(repaired, max_iters=20000)
        if sa_size > best_size and is_valid(sa_mask):
            best_mask = sa_mask
            best_size = sa_size
        
        # Another round of conflict repair on SA result
        if time.time() > deadline - 10:
            break
        repaired2 = conflict_repair(sa_mask)
        rep2_size = bin(repaired2).count('1')
        if rep2_size > best_size and is_valid(repaired2):
            best_mask = repaired2
            best_size = rep2_size
    
    # Final polish: try to add any remaining elements
    if time.time() < deadline - 3 and best_size > 0:
        for z in range(1, N + 1):
            if not (best_mask >> z) & 1:
                if can_add(z, best_mask):
                    best_mask |= (1 << z)
        final_size = bin(best_mask).count('1')
        if final_size > best_size:
            best_size = final_size
    
    # Convert bitmask to list
    result = []
    for z in range(1, N + 1):
        if (best_mask >> z) & 1:
            result.append(z)

    if not result:
        result = [1]

    return result
