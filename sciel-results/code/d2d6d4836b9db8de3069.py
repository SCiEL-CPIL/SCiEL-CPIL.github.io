import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n = 16
    triples = []
    triple_indices = {}
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triple_indices[(a, b, c)] = len(triples)
                triples.append((a, b, c))
    num_triples = len(triples)

    constraint_to_triples = []
    triple_to_constraints = [[] for _ in range(num_triples)]

    def get_idx(a, b, c):
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        return triple_indices[(a, b, c)]

    for v1 in range(n):
        for v2 in range(v1 + 1, n):
            for v3 in range(v2 + 1, n):
                for v4 in range(v3 + 1, n):
                    for v5 in range(v4 + 1, n):
                        for v6 in range(v5 + 1, n):
                            verts = [v1, v2, v3, v4, v5, v6]
                            for j in range(1, 6):
                                pair1 = (verts[0], verts[j])
                                remaining = [v for i, v in enumerate(verts) if i != 0 and i != j]
                                pairings = [
                                    [(remaining[0], remaining[1]), (remaining[2], remaining[3])],
                                    [(remaining[0], remaining[2]), (remaining[1], remaining[3])],
                                    [(remaining[0], remaining[3]), (remaining[1], remaining[2])]
                                ]
                                for p in pairings:
                                    pair2 = p[0]
                                    pair3 = p[1]
                                    t_indices = []
                                    for a in pair1:
                                        for b in pair2:
                                            for c in pair3:
                                                t_idx = get_idx(a, b, c)
                                                t_indices.append(t_idx)
                                                triple_to_constraints[t_idx].append(len(constraint_to_triples))
                                    constraint_to_triples.append(t_indices)

    num_constraints = len(constraint_to_triples)

    # Precompute for efficiency
    constraint_size = np.array([len(ct) for ct in constraint_to_triples], dtype=np.int32)

    def evaluate_solution(present):
        """Count violations and return (size, num_violations)"""
        counts = np.zeros(num_constraints, dtype=np.int32)
        for c in range(num_constraints):
            ct = constraint_to_triples[c]
            cnt = 0
            for t in ct:
                if present[t]:
                    cnt += 1
            counts[c] = cnt
        size = int(np.sum(present))
        violations = int(np.sum(counts >= 8))
        return size, violations, counts

    def greedy_resolve(present, counts):
        """Remove triples to fix violations. Returns (present, counts, removed_list)"""
        removed = []
        is_violated = counts >= 8
        violated_list = [c for c in range(num_constraints) if is_violated[c]]
        
        while violated_list:
            if time.time() > deadline - 15:
                break
            # Pick violated constraint with most remaining triples
            best_c = -1
            best_remaining = -1
            for c in violated_list:
                if not is_violated[c]:
                    continue
                rem = 0
                for t in constraint_to_triples[c]:
                    if present[t]:
                        rem += 1
                if rem > best_remaining:
                    best_remaining = rem
                    best_c = c
            
            if best_c == -1:
                # Clean up list
                violated_list = [c for c in violated_list if is_violated[c]]
                continue
            
            c = best_c
            # Find triple in this constraint that appears in most other violated constraints
            best_t = -1
            best_score = -1
            for t in constraint_to_triples[c]:
                if not present[t]:
                    continue
                score = 0
                for c2 in triple_to_constraints[t]:
                    if is_violated[c2]:
                        score += 1
                score += random.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_t = t
            
            if best_t == -1:
                violated_list = [c for c in violated_list if is_violated[c]]
                continue
            
            present[best_t] = False
            removed.append(best_t)
            for c2 in triple_to_constraints[best_t]:
                counts[c2] -= 1
                if counts[c2] == 7 and is_violated[c2]:
                    is_violated[c2] = False
        
        return present, counts, removed

    def local_search(present, counts, removed, max_steps=30000):
        """Local search: try to add back removed triples, or swap"""
        size = int(np.sum(present))
        is_violated = counts >= 8
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if not removed:
                break
            
            # Try adding a random removed triple
            ri = random.randint(0, len(removed) - 1)
            t = removed[ri]
            
            # Check if adding t would create violations
            new_violations = 0
            for c2 in triple_to_constraints[t]:
                if counts[c2] == 7:
                    new_violations += 1
            
            if new_violations == 0:
                # Safe add
                present[t] = True
                size += 1
                removed[ri] = removed[-1]
                removed.pop()
                for c2 in triple_to_constraints[t]:
                    counts[c2] += 1
                    if counts[c2] == 8:
                        is_violated[c2] = True
            else:
                # Try swap: remove one triple from the violated constraints, add t
                # Find a triple in the newly violated constraints that we can remove
                for c2 in triple_to_constraints[t]:
                    if counts[c2] == 7:
                        # Find a triple in c2 to remove
                        candidates = [t2 for t2 in constraint_to_triples[c2] if present[t2] and t2 != t]
                        if candidates:
                            # Pick the one that appears in fewest other constraints
                            best_t2 = candidates[0]
                            best_overlap = float('inf')
                            for t2 in candidates[:min(10, len(candidates))]:
                                ov = len(triple_to_constraints[t2])
                                if ov < best_overlap:
                                    best_overlap = ov
                                    best_t2 = t2
                            
                            # Check if removing t2 and adding t is net positive
                            # t2 removal might fix other violations
                            fixed = 0
                            for c3 in triple_to_constraints[t2]:
                                if counts[c3] == 8:
                                    fixed += 1
                            
                            if fixed >= new_violations:
                                # Do the swap
                                present[t2] = False
                                present[t] = True
                                removed[ri] = t2
                                removed.pop()
                                # Update counts
                                for c3 in triple_to_constraints[t2]:
                                    counts[c3] -= 1
                                    if counts[c3] == 7 and is_violated[c3]:
                                        is_violated[c3] = False
                                for c3 in triple_to_constraints[t]:
                                    counts[c3] += 1
                                    if counts[c3] == 8:
                                        is_violated[c3] = True
                                break
        
        return present, counts, removed, size

    def structured_init(seed):
        """Generate a structurally diverse initial solution"""
        random.seed(seed)
        
        # Different structural strategies
        strategy = seed % 4
        
        present = np.ones(num_triples, dtype=bool)
        
        if strategy == 0:
            # Random subset
            for i in range(num_triples):
                if random.random() < 0.6:
                    present[i] = False
        elif strategy == 1:
            # Cyclic pattern: keep triples where (a+b+c) % k < threshold
            k = 4 + (seed % 4)
            for i, (a, b, c) in enumerate(triples):
                if (a + b + c) % k >= k // 2:
                    present[i] = False
        elif strategy == 2:
            # Partition-based: split into 4 groups of 4, keep intra-group triples
            groups = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
            group_of = {}
            for gi, g in enumerate(groups):
                for v in g:
                    group_of[v] = gi
            for i, (a, b, c) in enumerate(triples):
                ga, gb, gc = group_of[a], group_of[b], group_of[c]
                if not (ga == gb == gc):
                    if random.random() < 0.3:
                        present[i] = False
        else:
            # Random with bias toward low-index points
            for i, (a, b, c) in enumerate(triples):
                bias = 1.0 - (a + b + c) / 45.0
                if random.random() > 0.5 + bias * 0.3:
                    present[i] = False
        
        # Compute counts
        counts = np.zeros(num_constraints, dtype=np.int32)
        for c in range(num_constraints):
            ct = constraint_to_triples[c]
            cnt = 0
            for t in ct:
                if present[t]:
                    cnt += 1
            counts[c] = cnt
        
        # Get removed list
        removed = [i for i in range(num_triples) if not present[i]]
        
        return present, counts, removed

    best_size = -1
    best_present = None

    num_restarts = 30

    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Structured initialization
        present, counts, removed = structured_init(restart * 7 + 3)
        
        # Greedy resolve violations
        present, counts, removed = greedy_resolve(present, counts)
        
        # Local search to add back
        present, counts, removed, size = local_search(present, counts, removed, max_steps=15000)
        
        if size > best_size:
            best_size = size
            best_present = present.copy()
        
        # Occasionally do a second round of local search with different parameters
        if time.time() < deadline - 10 and size > best_size * 0.95:
            present2, counts2, removed2, size2 = local_search(present.copy(), counts.copy(), removed.copy(), max_steps=10000)
            if size2 > best_size:
                best_size = size2
                best_present = present2.copy()

    if best_size == -1:
        return []

    result = []
    for i in range(num_triples):
        if best_present[i]:
            result.append(list(triples[i]))

    return result
