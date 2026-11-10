import time
import random
from itertools import combinations

N = 256
DEADLINE = 85.0  # Safety margin before 90s limit


def _count_3ap(selected_set):
    """Count 3-APs in a set."""
    count = 0
    sel = selected_set
    for a in sel:
        for b in sel:
            if b > a:
                c = 2 * b - a
                if c in sel and c > b:
                    count += 1
    return count


def _has_4ap(selected_set, new_elem):
    """Check if adding new_elem to selected_set creates a 4-AP."""
    sel = selected_set
    d_max = (N - 1) // 3
    
    # new_elem as first: a, a+d, a+2d, a+3d
    for d in range(1, d_max + 1):
        if new_elem + d in sel and new_elem + 2*d in sel and new_elem + 3*d in sel:
            return True
    # new_elem as second: a, a+d, a+2d, a+3d where a+d=new_elem
    for d in range(1, d_max + 1):
        a = new_elem - d
        if a >= 1 and a in sel and new_elem + d in sel and new_elem + 2*d in sel:
            return True
    # new_elem as third: a, a+d, a+2d, a+3d where a+2d=new_elem
    for d in range(1, d_max + 1):
        a = new_elem - 2*d
        if a >= 1 and a in sel and new_elem - d in sel and new_elem + d in sel:
            return True
    # new_elem as fourth: a, a+d, a+2d, a+3d where a+3d=new_elem
    for d in range(1, d_max + 1):
        a = new_elem - 3*d
        if a >= 1 and a in sel and a + d in sel and a + 2*d in sel:
            return True
    return False


def _marginal_3ap_gain(selected_set, x):
    """Count how many new 3-APs are created by adding x to selected_set."""
    gain = 0
    sel = selected_set
    # x as first: x, b, c where b, c in sel and c = 2b - x
    for b in sel:
        if b > x:
            c = 2 * b - x
            if c in sel and c > b:
                gain += 1
    # x as second: a, x, c where a, c in sel and c = 2x - a
    for a in sel:
        if a < x:
            c = 2 * x - a
            if c in sel and c > x:
                gain += 1
    # x as third: a, b, x where a, b in sel and b = (a + x) / 2
    for a in sel:
        if a < x:
            b = (a + x) // 2
            if (a + x) % 2 == 0 and b in sel and a < b < x:
                gain += 1
    return gain


def _density_greedy(start_time):
    """Greedy: at each step, add the element with max marginal 3-AP gain."""
    selected = set()
    remaining = set(range(1, N + 1))
    
    while remaining:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        best_elem = None
        best_gain = -1
        best_score = 0
        
        for x in remaining:
            if _has_4ap(selected, x):
                continue
            gain = _marginal_3ap_gain(selected, x)
            if gain > best_gain:
                best_gain = gain
                best_elem = x
                best_score = gain
        
        if best_elem is None:
            break
        
        selected.add(best_elem)
        remaining.discard(best_elem)
    
    return selected


def _simulated_annealing(selected, start_time, max_iters=50000):
    """Simulated annealing to polish the set."""
    current = set(selected)
    current_score = _count_3ap(current)
    best_set = set(current)
    best_score = current_score
    
    T = 5.0  # Initial temperature
    cooling = 0.9995
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Try adding an element
            candidates = [x for x in range(1, N + 1) if x not in current]
            if not candidates:
                continue
            x = random.choice(candidates)
            if _has_4ap(current, x):
                continue
            new_set = current | {x}
            new_score = _count_3ap(new_set)
        elif move_type < 0.6:
            # Try removing an element
            if not current:
                continue
            x = random.choice(list(current))
            new_set = current - {x}
            new_score = _count_3ap(new_set)
        else:
            # Try swapping: remove one, add another
            if not current:
                continue
            x = random.choice(list(current))
            candidates = [y for y in range(1, N + 1) if y not in current and y != x]
            if not candidates:
                continue
            y = random.choice(candidates)
            if _has_4ap(current - {x}, y):
                continue
            new_set = (current - {x}) | {y}
            new_score = _count_3ap(new_set)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < (2.718281828 ** (delta / T)):
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = set(current)
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best_set, best_score


def _multi_start_greedy(start_time, n_starts=5):
    """Try multiple density-greedy starts with slight random perturbations."""
    best_set = set()
    best_score = 0
    
    for s in range(n_starts):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Slight randomization: start with a small random seed set
        seed_set = set()
        if s > 0:
            # Add 1-3 random elements as seed
            n_seed = min(3, N)
            seed_elements = random.sample(range(1, N + 1), n_seed)
            for x in seed_elements:
                if not _has_4ap(seed_set, x):
                    seed_set.add(x)
        
        # Run density greedy from this seed
        selected = set(seed_set)
        remaining = set(range(1, N + 1)) - selected
        
        while remaining:
            if time.time() - start_time > DEADLINE - 20:
                break
            
            best_elem = None
            best_gain = -1
            
            for x in remaining:
                if _has_4ap(selected, x):
                    continue
                gain = _marginal_3ap_gain(selected, x)
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
            
            if best_elem is None:
                break
            
            selected.add(best_elem)
            remaining.discard(best_elem)
        
        score = _count_3ap(selected)
        if score > best_score:
            best_score = score
            best_set = selected
    
    return best_set, best_score


def construct():
    start_time = time.time()
    random.seed(123)
    
    # Phase 1: Multi-start density greedy
    best_set, best_score = _multi_start_greedy(start_time, n_starts=3)
    
    # Phase 2: Simulated annealing polish
    if time.time() - start_time < DEADLINE - 10:
        best_set, best_score = _simulated_annealing(best_set, start_time, max_iters=30000)
    
    # Phase 3: Another annealing pass with different temperature schedule
    if time.time() - start_time < DEADLINE - 5:
        best_set2, best_score2 = _simulated_annealing(best_set, start_time, max_iters=20000)
        if best_score2 > best_score:
            best_set = best_set2
            best_score = best_score2
    
    return sorted(best_set)
