import time
import random
import math
from itertools import combinations

def score_set(U):
    """Compute the score for a given set U."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute U+U
    sum_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
    
    # Compute U-U
    diff_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Start with a geometric-like seed
    def make_geometric_seed(n, base):
        return [base**i for i in range(n)]
    
    # Try various seeds
    seeds = []
    for n in range(4, 20):
        for base in [2, 3, 5, 10, 20]:
            if base**(n-1) > 10**7:
                break
            s = make_geometric_seed(n, base)
            seeds.append(s)
    
    # Also try some random sparse sets
    for _ in range(50):
        n = random.randint(5, 25)
        max_val = random.randint(100, 10000)
        s = [0] + sorted(random.sample(range(1, max_val), n-1))
        seeds.append(s)
    
    # Evaluate seeds
    for s in seeds:
        if time.time() > deadline - 30:
            break
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)
    
    # Local search: try adding, removing, or perturbing elements
    def local_search(U, max_iters=500):
        U = sorted(set(U))
        current_score = score_set(U)
        best_local = list(U)
        best_local_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Try random modifications
            move_type = random.choice(['add', 'remove', 'perturb', 'add_perturb'])
            
            if move_type == 'add' and len(U) < 30:
                # Add a new element
                max_u = max(U) if U else 0
                new_elem = random.randint(1, max(10, max_u * 2))
                new_U = sorted(set(U + [new_elem]))
            elif move_type == 'remove' and len(U) > 3:
                # Remove a random non-zero element
                candidates = [x for x in U if x != 0]
                if candidates:
                    rem = random.choice(candidates)
                    new_U = sorted(set(U - {rem}))
                else:
                    continue
            elif move_type == 'perturb':
                # Perturb a random non-zero element
                candidates = [x for x in U if x != 0]
                if not candidates:
                    continue
                idx = random.randint(1, len(U)-1)
                old = U[idx]
                delta = random.randint(-max(1, old//10), max(1, old//10))
                new_val = old + delta
                if new_val < 0:
                    new_val = old - delta
                if new_val == 0:
                    new_val = 1
                new_U = sorted(set(list(U[:idx]) + [new_val] + list(U[idx+1:])))
            else:  # add_perturb
                # Replace one element with a nearby one
                candidates = [x for x in U if x != 0]
                if not candidates:
                    continue
                idx = random.randint(1, len(U)-1)
                old = U[idx]
                new_val = old + random.randint(-max(5, old//5), max(5, old//5))
                if new_val < 0:
                    new_val = old - random.randint(1, max(5, old//5))
                if new_val == 0:
                    new_val = 1
                new_U = sorted(set(list(U[:idx]) + [new_val] + list(U[idx+1:])))
            
            if len(new_U) != len(U) and move_type not in ['add', 'remove']:
                continue
            
            new_score = score_set(new_U)
            if new_score > best_local_score:
                best_local_score = new_score
                best_local = list(new_U)
        
        return best_local, best_local_score
    
    # Run local search from best seed
    if best_set:
        improved, improved_score = local_search(best_set, max_iters=1000)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved
    
    # Try multiple restarts with different seeds
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        # Generate a new random seed
        n = random.randint(5, 20)
        max_val = random.randint(50, 5000)
        s = [0] + sorted(random.sample(range(1, max_val), min(n-1, max_val-1)))
        
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)
        
        # Local search from this seed
        improved, improved_score = local_search(s, max_iters=300)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved
    
    # Final polish: try to optimize the current best
    if time.time() < deadline - 5 and best_set:
        # More careful local search
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            improved, improved_score = local_search(best_set, max_iters=500)
            if improved_score > best_score:
                best_score = improved_score
                best_set = improved
            else:
                break
    
    return sorted(set(best_set))
