import time
from math import log, floor, sqrt
import random
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def score_set_fast(U):
    """Faster scoring using numpy for larger sets"""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U)
    # Compute all pairwise sums and differences
    sums = np.add.outer(U_arr, U_arr)
    diffs = np.subtract.outer(U_arr, U_arr)
    
    sum_count = len(np.unique(sums))
    diff_count = len(np.unique(diffs))
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)

def genetic_algorithm(U, deadline, pop_size=20, generations=50):
    """
    Genetic algorithm treating set membership as genome.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    # Determine search space
    max_val = max(U) if U else 10
    search_max = max_val * 3 + 50
    
    # Create initial population
    population = [list(U)]
    for _ in range(pop_size - 1):
        # Mutate: add/remove/replace elements
        mutant = list(U)
        for _ in range(random.randint(1, 3)):
            op = random.choice(['add', 'remove', 'replace'])
            if op == 'add' and len(mutant) < 30:
                val = random.randint(1, search_max)
                if val not in mutant:
                    mutant.append(val)
            elif op == 'remove' and len(mutant) > 2:
                removable = [x for x in mutant if x != 0]
                if removable:
                    mutant.remove(random.choice(removable))
            elif op == 'replace' and len(mutant) > 1:
                idx = random.randint(1, len(mutant) - 1)
                mutant[idx] = random.randint(1, search_max)
        mutant = sorted(set(mutant))
        if 0 in mutant and len(mutant) >= 2:
            population.append(mutant)
    
    for gen in range(generations):
        if time.time() > deadline - 5:
            break
        
        # Score population
        scored = []
        for ind in population:
            s = score_set(ind)
            scored.append((s, ind))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_set = list(scored[0][1])
        
        # Selection: keep top half
        elite = [ind for s, ind in scored[:pop_size//2]]
        
        # Generate new population
        new_population = list(elite)
        while len(new_population) < pop_size:
            if random.random() < 0.3 and len(elite) >= 2:
                # Crossover
                p1, p2 = random.sample(elite, 2)
                # Take elements from both parents
                child = list(set(p1 + p2))
                if 0 not in child:
                    child.append(0)
                child = sorted(set(child))[:30]
            else:
                # Mutation from random parent
                parent = random.choice(elite)
                child = list(parent)
                for _ in range(random.randint(1, 4)):
                    op = random.choice(['add', 'remove', 'replace', 'perturb'])
                    if op == 'add' and len(child) < 30:
                        val = random.randint(1, search_max)
                        if val not in child:
                            child.append(val)
                    elif op == 'remove' and len(child) > 2:
                        removable = [x for x in child if x != 0]
                        if removable:
                            child.remove(random.choice(removable))
                    elif op == 'replace' and len(child) > 1:
                        idx = random.randint(1, len(child) - 1)
                        child[idx] = random.randint(1, search_max)
                    elif op == 'perturb' and len(child) > 1:
                        idx = random.randint(1, len(child) - 1)
                        child[idx] = max(1, child[idx] + random.randint(-5, 5))
                child = sorted(set(child))
                if 0 not in child:
                    child.append(0)
                child = sorted(set(child))[:30]
            
            if len(child) >= 2 and 0 in child:
                new_population.append(child)
        
        population = new_population
    
    return best_score, best_set

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.15:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2 + 50)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.4:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try more random candidates for better exploration
                random.shuffle(candidates)
                for c in candidates[:20]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2 + 50))
                        break
        elif move_type < 0.6:
            # Try to REMOVE an element
            if len(U) > 2:
                # Try multiple removals to find the best
                removable = [x for x in U if x != 0]
                best_remove_score = -1
                best_remove_set = None
                for c in removable[:10]:
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > best_remove_score:
                        best_remove_score = new_score
                        best_remove_set = new_U
                if best_remove_set is not None and best_remove_score > current_score:
                    U = best_remove_set
                    current_score = best_remove_score
                    improved = True
                    candidates = generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Pick a random candidate
                random.shuffle(candidates)
                for new_val in candidates[:20]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2 + 50))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates (from Inspiration)
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Geometric Search (from Current)
    # Quick coarse grid to find a good geometric seed
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 50:
            break
        for m in range(2, 21):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 3. Dynamic Local Search (from Inspiration)
    # Use the best found so far as the primary seed, and structured ones as secondary seeds
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 30:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        # Run local search
        local_score, local_set = local_search_dynamic(U, deadline - 10, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts? 
            # To keep it simple, we just update the primary seed list
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 4. Genetic Algorithm Refinement
    if time.time() < deadline - 15:
        ga_score, ga_set = genetic_algorithm(list(best_set), deadline - 5, pop_size=15, generations=30)
        if ga_score > best_score:
            best_score = ga_score
            best_set = ga_set

    # 5. Final Polish: Fine-grained local search on the best set
    # This addresses the "refinement machinery" requirement.
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 800:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values (small delta)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try deltas in a wider range but smaller steps
                for delta in range(-15, 16):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element near the max
            if not improved and n < 25 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max_val - 15, max_val + 16):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

# Ensure 0 is in the set and it's a list of nonnegative integers
# The function construct returns a list.
# The evaluator calls construct().

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
