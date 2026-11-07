import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    def is_comparable(a, b):
        return (a & ~b) == 0 or (b & ~a) == 0

    # Precompute neighbors for each mask
    # neighbors[mask] is a list of masks comparable to mask (excluding itself)
    neighbors = [[] for _ in range(total_masks)]
    
    for a in range(total_masks):
        # Subsets of a
        sub = a
        while True:
            if sub != a:
                neighbors[a].append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & a
        
        # Supersets of a
        comp = (total_masks - 1) ^ a
        c = comp
        while True:
            b = a | c
            if b != a:
                neighbors[a].append(b)
            if c == 0:
                break
            c = (c - 1) & comp

    def calc_initial_deg(family_set):
        deg = [0] * total_masks
        for a in family_set:
            for nb in neighbors[a]:
                deg[nb] += 1
        return deg

    def hill_climb(start_family):
        family_set = set(start_family)
        deg = calc_initial_deg(family_set)
        
        improved = True
        T = 1.0
        cooling = 0.999
        
        while time.time() < deadline:
            improved = False
            best_delta = 0
            best_a = None
            best_b = None
            
            # Find best single swap
            for a in family_set:
                for b in range(total_masks):
                    if b in family_set:
                        continue
                    
                    is_comp = 1 if is_comparable(a, b) else 0
                    delta = (deg[b] - is_comp) - deg[a]
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_delta > 0:
                # Apply move
                family_set.remove(best_a)
                family_set.add(best_b)
                
                # Update deg
                for nb in neighbors[best_a]:
                    deg[nb] -= 1
                for nb in neighbors[best_b]:
                    deg[nb] += 1
                    
                improved = True
            else:
                # No improvement. Try to escape local optimum.
                # Pick a random move
                if random.random() < 0.5:
                    a = random.choice(list(family_set))
                    b = random.randint(0, total_masks - 1)
                    while b in family_set:
                        b = random.randint(0, total_masks - 1)
                    
                    is_comp = 1 if is_comparable(a, b) else 0
                    delta = (deg[b] - is_comp) - deg[a]
                    
                    # Accept if delta is not too bad, or with probability exp(-delta/T)
                    # Since delta <= 0, -delta >= 0.
                    if random.random() < pow(2.71828, delta / T):
                        family_set.remove(a)
                        family_set.add(b)
                        for nb in neighbors[a]:
                            deg[nb] -= 1
                        for nb in neighbors[b]:
                            deg[nb] += 1
                        improved = True
                        T *= cooling
                else:
                    # Cool down
                    T *= cooling
                    if T < 0.01:
                        break

        return list(family_set)

    # Generate starts
    starts = []
    
    def greedy_construct():
        family = []
        used = set()
        family.append(0)
        used.add(0)
        
        all_masks = list(range(1, total_masks))
        random.shuffle(all_masks)
        
        while len(family) < m:
            best_gain = -1
            best_mask = None
            for s in all_masks:
                if s in used:
                    continue
                gain = 0
                for f in family:
                    if is_comparable(s, f):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            if best_mask is None:
                break
            family.append(best_mask)
            used.add(best_mask)
            
        return family

    starts.append(greedy_construct())
    
    for _ in range(3):
        starts.append(random.sample(range(total_masks), m))

    best_family = None
    best_score = -1

    def calc_score(fam):
        s = 0
        L = len(fam)
        for i in range(L):
            ai = fam[i]
            for j in range(i+1, L):
                if is_comparable(ai, fam[j]):
                    s += 1
        return s

    for start in starts:
        if time.time() > deadline - 2:
            break
        fam = hill_climb(start)
        score = calc_score(fam)
        if score > best_score:
            best_score = score
            best_family = fam

    if best_family is None:
        best_family = starts[0]

    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
