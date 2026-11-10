"""Reciprocal weight of three-progression-free sets via randomized local search."""
import random
import time
from fractions import Fraction

N = 256
DEADLINE = 85.0  # safety margin before 90s kill

def _is_3free(S):
    """Check if set S has no nonconstant 3-term AP."""
    Sset = set(S)
    for a in S:
        for b in S:
            if a == b:
                continue
            c = 2 * b - a
            if c != b and c in Sset:
                return False
    return True

def _conflicts_with(x, S):
    """Return list of (a, b) in S such that x, a, b form a 3-term AP."""
    Sset = set(S)
    conflicts = []
    for a in S:
        # x, a, b: b = 2a - x
        b = 2 * a - x
        if b != a and b != x and b in Sset:
            conflicts.append((a, b))
        # a, x, b: b = 2x - a
        b = 2 * x - a
        if b != a and b != x and b in Sset:
            conflicts.append((a, b))
        # a, b, x: b = (a + x) / 2, must be integer
        if (a + x) % 2 == 0:
            b = (a + x) // 2
            if b != a and b != x and b in Sset:
                conflicts.append((a, b))
    return conflicts

def _can_add(x, S):
    """Check if x can be added to S without creating a 3-term AP."""
    Sset = set(S)
    for a in S:
        for b in S:
            if a == b:
                continue
            # Check if x, a, b or a, x, b or a, b, x forms AP
            # x = 2b - a (x is the end)
            if 2 * b - a == x:
                return False
            # x = 2a - b (x is the start)  
            if 2 * a - b == x:
                return False
            # b = 2x - a (x is the middle) -- but this means a, x, b with b in S
            # Actually simpler: for each pair (a,b) in S, check if x completes an AP
            # AP: a, b, c where c = 2b-a; or a, c, b where c = 2a-b; or c, a, b where c = 2a-b... 
            # Let's just check all three positions:
            # If x is first: x, a, b => 2a - x = b => x = 2a - b
            # If x is middle: a, x, b => 2x = a + b => x = (a+b)/2
            # If x is last: a, b, x => 2b - a = x
            if 2 * a - x == b:
                return False
            if (a + b) % 2 == 0 and (a + b) // 2 == x:
                return False
            if 2 * b - a == x:
                return False
    return True

def _score(S):
    """Exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total

def _random_3free_init(rng):
    """Build a random 3-free set by random ordering."""
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    S = set()
    for x in candidates:
        if _can_add(x, S):
            S.add(x)
    return S

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_set = None
    best_score = Fraction(0, 1)
    
    # Initial solution: random 3-free set
    S = _random_3free_init(rng)
    current_score = _score(S)
    best_set = set(S)
    best_score = current_score
    
    # Simulated annealing / hill climbing
    temperature = 1.0
    cooling = 0.999
    min_temp = 0.001
    
    while time.time() - start_time < DEADLINE:
        if temperature < min_temp:
            # Reset for another round
            temperature = 1.0
            # Perturb best solution
            S = set(best_set)
            # Remove a random small element to create space
            small_elements = sorted([x for x in S if x <= 50])
            if small_elements:
                to_remove = rng.choice(small_elements)
                S.remove(to_remove)
            current_score = _score(S)
        
        # Try a random move
        move_type = rng.random()
        if move_type < 0.6:
            # Add a random element not in S
            available = [x for x in range(1, N + 1) if x not in S]
            if not available:
                continue
            x = rng.choice(available)
            if _can_add(x, S):
                new_S = S | {x}
                delta = Fraction(1, x)
                if current_score + delta > best_score:
                    best_set = new_S
                    best_score = current_score + delta
                    S = new_S
                    current_score += delta
                elif rng.random() < temperature:
                    S = new_S
                    current_score += delta
            else:
                # Can't add directly; try removing conflicting elements
                conflicts = _conflicts_with(x, S)
                if conflicts and len(conflicts) <= 2:
                    # Try removing one conflicting element
                    a, b = rng.choice(conflicts)
                    # Remove the smaller of a, b (keep larger for better reciprocal weight)
                    to_remove = min(a, b)
                    if to_remove in S:
                        new_S = (S - {to_remove}) | {x}
                        if _is_3free(new_S):
                            delta = Fraction(1, x) - Fraction(1, to_remove)
                            new_score = current_score + delta
                            if new_score > best_score:
                                best_set = new_S
                                best_score = new_score
                                S = new_S
                                current_score = new_score
                            elif rng.random() < temperature:
                                S = new_S
                                current_score = new_score
        elif move_type < 0.8:
            # Remove a random small element
            small_elements = sorted([x for x in S if x <= 30])
            if small_elements:
                x = rng.choice(small_elements)
                new_S = S - {x}
                delta = -Fraction(1, x)
                new_score = current_score + delta
                if rng.random() < temperature:
                    S = new_S
                    current_score = new_score
        else:
            # Swap: remove small, try to add large
            small_elements = sorted([x for x in S if x <= 40])
            large_candidates = [x for x in range(100, N + 1) if x not in S]
            if small_elements and large_candidates:
                x_remove = rng.choice(small_elements)
                x_add = rng.choice(large_candidates)
                temp_S = S - {x_remove}
                if _can_add(x_add, temp_S):
                    new_S = temp_S | {x_add}
                    delta = Fraction(1, x_add) - Fraction(1, x_remove)
                    new_score = current_score + delta
                    if new_score > best_score:
                        best_set = new_S
                        best_score = new_score
                        S = new_S
                        current_score = new_score
                    elif rng.random() < temperature:
                        S = new_S
                        current_score = new_score
        
        temperature *= cooling
    
    # Final: try greedy improvement on best_set
    S = set(best_set)
    improved = True
    while improved and time.time() - start_time < DEADLINE - 2:
        improved = False
        for x in range(1, N + 1):
            if x not in S and _can_add(x, S):
                S.add(x)
                improved = True
                break
    
    return sorted(S)
