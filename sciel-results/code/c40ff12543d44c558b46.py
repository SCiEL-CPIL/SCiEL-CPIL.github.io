import time
import random
import numpy as np

_START = time.time()
_DEADLINE = 55.0  # safety margin before 60s kill

def _sq_nonzero_count(coefs):
    """Count nonzero terms in the exact square using numpy convolution."""
    arr = np.array(coefs, dtype=np.int64)
    sq = np.convolve(arr, arr)
    return int(np.count_nonzero(sq))

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _construct_from_factors(rng):
    """
    Structural approach: try to build a 12-term polynomial P such that P^2 is sparse.
    Key idea: if P = A + B where A and B have disjoint supports and cross terms cancel,
    or P has a multiplicative structure.
    
    More directly: try palindromic polynomials with coefficients from a small set,
    and check if the square has few nonzeros. Use the fact that for palindromic P,
    P^2 is also palindromic, halving the effective search space.
    """
    best = [10**9, None]
    
    # Try palindromic structures: P(x) = sum a_i x^i where a_i = a_{20-i}
    # This means 6 independent coefficients (plus possibly center term)
    # For degree 40 palindromic: a_0..a_20 with a_i = a_{40-i}
    # That's 21 coefficients but only ~11 independent + center
    
    # Alternative: try P with exactly 12 terms in a palindromic pattern
    # e.g., terms at positions {d1, d2, ..., d6, 40-d6, ..., 40-d1}
    
    for _attempt in range(3000):
        if time.time() > _DEADLINE:
            break
        
        # Choose 6 positions for the "half" (positions 0-20)
        half_pos = sorted(rng.sample(range(0, 21), 6))
        # Ensure 40 is in the full support
        if 20 not in half_pos:
            # Replace one with 20 to ensure center, or ensure 40-d for some d
            pass
        
        full_support = set()
        for d in half_pos:
            full_support.add(d)
            if 40 - d != d:
                full_support.add(40 - d)
        
        if len(full_support) != 12:
            # Adjust: we need exactly 12, so we need 6 pairs with no fixed points
            # or 5 pairs + 2 fixed points (but 20 is only fixed point in 0-40)
            # So we need 6 pairs: d and 40-d for d in 0..19, none equal to 20
            continue
        
        support = sorted(full_support)
        if support[-1] != 40:
            continue
        
        # Assign coefficients: palindromic means coef[d] = coef[40-d]
        # Try small integer values
        coefs = [0] * 41
        vals = [rng.choice([-4,-3,-2,-1,1,2,3,4]) for _ in range(6)]
        
        for i, d in enumerate(half_pos):
            coefs[d] = vals[i]
            if 40 - d != d:
                coefs[40 - d] = vals[i]
        
        if coefs[40] == 0:
            coefs[40] = 1
        
        sc = _sq_nonzero_count(coefs)
        if sc < best[0]:
            best[0] = sc
            best[1] = coefs[:]
        
        # Local perturbation
        if best[1] is not None:
            cand = best[1][:]
            idx = rng.randrange(41)
            old = cand[idx]
            new = old + rng.choice([-3,-2,-1,1,2,3])
            if -64 <= new <= 64 and new != 0:
                cand[idx] = new
            # Maintain palindromic constraint
            if 40 - idx != idx:
                cand[40 - idx] = new
            if cand[40] == 0:
                cand[40] = 1
                if 40 - 40 != 40:
                    pass
            sc2 = _sq_nonzero_count(cand)
            if sc2 < best[0]:
                best[0] = sc2
                best[1] = cand[:]
    
    return best

def _genetic_search(rng):
    """
    Genetic algorithm on the coefficient vector directly.
    Population of 12-term polynomials, fitness = nonzero count in square.
    """
    pop_size = 40
    best = [10**9, None]
    
    def _init_individual():
        """Create a random valid 12-term polynomial."""
        # Choose 12 distinct positions including 40
        positions = sorted(rng.sample(range(0, 40), 11) + [40])
        coefs = [0] * 41
        for p in positions:
            coefs[p] = rng.choice([-4,-3,-2,-1,1,2,3,4])
        if coefs[40] == 0:
            coefs[40] = 1
        return coefs
    
    def _fitness(coefs):
        return _sq_nonzero_count(coefs)
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        ind = _init_individual()
        pop_score = _fitness(ind)
        population.append((pop_score, ind))
        if pop_score < best[0]:
            best[0] = pop_score
            best[1] = ind[:]
    
    # Evolution loop
    for gen in range(200):
        if time.time() > _DEADLINE:
            break
        
        # Sort by fitness
        population.sort(key=lambda x: x[0])
        
        # Keep top half
        survivors = population[:pop_size//2]
        
        # Generate offspring
        new_pop = []
        for _ in range(pop_size//2):
            if time.time() > _DEADLINE:
                break
            
            # Crossover: pick two parents, mix their nonzero positions
            p1 = survivors[rng.randrange(len(survivors))][1]
            p2 = survivors[rng.randrange(len(survivors))][1]
            
            # Create child by taking nonzero positions from both
            child = [0] * 41
            nz1 = [i for i in range(41) if p1[i] != 0]
            nz2 = [i for i in range(41) if p2[i] != 0]
            
            # Take 12 positions from union
            all_pos = sorted(set(nz1 + nz2))
            if len(all_pos) < 12:
                # Fill with random positions
                remaining = [i for i in range(40) if i not in all_pos]
                rng.shuffle(remaining)
                all_pos = sorted(all_pos + remaining[:12-len(all_pos)])
            
            chosen = rng.sample(all_pos, 12) if len(all_pos) >= 12 else all_pos + [40]
            chosen = sorted(set(chosen))
            if len(chosen) != 12:
                chosen = sorted(rng.sample(range(0, 40), 11) + [40])
            
            for p in chosen:
                # Inherit value from parent if available, else random
                if p1[p] != 0:
                    child[p] = p1[p]
                elif p2[p] != 0:
                    child[p] = p2[p]
                else:
                    child[p] = rng.choice([-4,-3,-2,-1,1,2,3,4])
            
            if child[40] == 0:
                child[40] = 1
            
            # Mutation
            for _ in range(rng.randrange(1, 4)):
                idx = rng.randrange(41)
                old = child[idx]
                new = old + rng.choice([-3,-2,-1,1,2,3])
                if -64 <= new <= 64 and new != 0:
                    child[idx] = new
                elif new == 0 and old != 0:
                    # Keep it nonzero to maintain 12 terms
                    pass
            
            # Ensure exactly 12 nonzero terms
            nz = sum(1 for c in child if c != 0)
            if nz != 12:
                # Fix: if too many, zero out some; if too few, add some
                nz_indices = [i for i in range(41) if child[i] != 0]
                if nz > 12:
                    to_zero = rng.sample(nz_indices, nz - 12)
                    if 40 in to_zero:
                        to_zero.remove(40)
                        to_zero.append(rng.choice([i for i in nz_indices if i != 40]))
                    for i in to_zero:
                        child[i] = 0
                elif nz < 12:
                    zero_indices = [i for i in range(40) if child[i] == 0]
                    rng.shuffle(zero_indices)
                    for i in zero_indices[:12-nz]:
                        child[i] = rng.choice([-4,-3,-2,-1,1,2,3,4])
            
            if child[40] == 0:
                child[40] = 1
            
            sc = _fitness(child)
            new_pop.append((sc, child))
            if sc < best[0]:
                best[0] = sc
                best[1] = child[:]
        
        population = survivors + new_pop
    
    return best

def _simulated_annealing(rng):
    """
    Simulated annealing on the coefficient vector.
    """
    best = [10**9, None]
    
    # Start from a good initial point
    positions = sorted(rng.sample(range(0, 40), 11) + [40])
    current = [0] * 41
    for p in positions:
        current[p] = rng.choice([-3,-2,-1,1,2,3])
    if current[40] == 0:
        current[40] = 1
    
    current_score = _sq_nonzero_count(current)
    best[0] = current_score
    best[1] = current[:]
    
    T = 10.0
    cooling = 0.995
    
    steps = 0
    while time.time() < _DEADLINE:
        steps += 1
        
        # Propose a move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Change a coefficient value
            nz_indices = [i for i in range(41) if current[i] != 0]
            idx = rng.choice(nz_indices)
            old = current[idx]
            new = old + rng.choice([-4,-3,-2,-1,1,2,3,4])
            if -64 <= new <= 64 and new != 0:
                current[idx] = new
                if current[40] == 0:
                    current[40] = 1
                new_score = _sq_nonzero_count(current)
                delta = new_score - current_score
                if delta <= 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
                    current_score = new_score
                    if new_score < best[0]:
                        best[0] = new_score
                        best[1] = current[:]
                else:
                    current[idx] = old
            else:
                current[idx] = old
        
        elif move_type < 0.7:
            # Swap two positions
            nz_indices = [i for i in range(41) if current[i] != 0]
            if len(nz_indices) >= 2:
                i, j = rng.sample(nz_indices, 2)
                current[i], current[j] = current[j], current[i]
                if current[40] == 0:
                    # Find a nonzero to put at 40
                    for k in range(40):
                        if current[k] != 0:
                            current[40] = current[k]
                            current[k] = 0
                            break
                new_score = _sq_nonzero_count(current)
                delta = new_score - current_score
                if delta <= 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
                    current_score = new_score
                    if new_score < best[0]:
                        best[0] = new_score
                        best[1] = current[:]
                else:
                    current[i], current[j] = current[j], current[i]
        
        else:
            # Change support: move a term to a new position
            nz_indices = [i for i in range(41) if current[i] != 0]
            if nz_indices:
                idx = rng.choice(nz_indices)
                if idx == 40:
                    # Don't move the last term
                    pass
                else:
                    new_pos = rng.randrange(0, 40)
                    if current[new_pos] == 0:
                        current[new_pos] = current[idx]
                        current[idx] = 0
                        new_score = _sq_nonzero_count(current)
                        delta = new_score - current_score
                        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
                            current_score = new_score
                            if new_score < best[0]:
                                best[0] = new_score
                                best[1] = current[:]
                        else:
                            current[idx] = current[new_pos]
                            current[new_pos] = 0
        
        T *= cooling
    
    return best

def _polish(best, rng):
    """Local hill-climbing polish."""
    best_local = best[1][:] if best[1] else None
    if best_local is None:
        return best
    best_score = best[0]
    
    val_candidates = [-64,-32,-16,-8,-4,-3,-2,-1,1,2,3,4,8,16,32,64]
    
    for _pass in range(10):
        if time.time() > _DEADLINE:
            break
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            nz_indices = [i for i in range(41) if best_local[i] != 0]
            for d in nz_indices:
                if time.time() > _DEADLINE:
                    break
                old_val = best_local[d]
                for new_val in val_candidates:
                    if new_val == old_val:
                        continue
                    best_local[d] = new_val
                    if best_local[40] == 0:
                        best_local[d] = old_val
                        continue
                    sc = _sq_nonzero_count(best_local)
                    if sc < best_score:
                        best_score = sc
                        best[0] = sc
                        best[1] = best_local[:]
                        improved = True
                        break
                    else:
                        best_local[d] = old_val
    return best

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 55.0
    rng = random.Random(12345)
    
    # Combine multiple structural approaches
    best = [10**9, None]
    
    # 1) Palindromic construction
    b1 = _construct_from_factors(rng)
    if b1[0] < best[0]:
        best = b1[:]
    
    # 2) Genetic algorithm
    if time.time() < _DEADLINE:
        b2 = _genetic_search(rng)
        if b2[0] < best[0]:
            best = b2[:]
    
    # 3) Simulated annealing
    if time.time() < _DEADLINE:
        b3 = _simulated_annealing(rng)
        if b3[0] < best[0]:
            best = b3[:]
    
    # 4) Polish the best
    if best[1] is not None:
        _polish(best, rng)
    
    if best[1] is None:
        coefs = [0] * 41
        for i in range(12):
            coefs[i] = 1
        return coefs
    
    return best[1]
