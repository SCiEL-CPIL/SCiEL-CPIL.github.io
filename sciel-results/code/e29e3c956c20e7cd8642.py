"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 128}

def _is_valid(S):
    """Check if set S has no three distinct members a,b,c with all pair sums in S."""
    s = set(S)
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            for k in range(j+1, n):
                c = lst[k]
                if (a+b) in s and (a+c) in s and (b+c) in s:
                    return False
    return True

def _count_violations(S, n=128):
    """Count the number of violating triples in S (optimized)."""
    s = set(S)
    lst = sorted(S)
    n_elem = len(lst)
    count = 0
    for i in range(n_elem):
        a = lst[i]
        # Collect b > a such that a+b in S and a+b > b (always true since a>0)
        # and a+b <= n
        candidates = []
        for j in range(i+1, n_elem):
            b = lst[j]
            ab = a + b
            if ab > n:
                break  # since lst is sorted, b only increases
            if ab in s:
                candidates.append((b, ab))
        # For each pair (b, c) with b < c in candidates, check if b+c in S
        for idx1 in range(len(candidates)):
            b = candidates[idx1][0]
            for idx2 in range(idx1+1, len(candidates)):
                c = candidates[idx2][0]
                bc = b + c
                if bc <= n and bc in s:
                    count += 1
    return count

def _violation_count_incremental(S, n=128):
    """Fast check: count violations involving a specific element x just added.
    Returns the number of new violations created by adding x to set S_old."""
    s = set(S_old) | {x}  # S_new
    # New violations must include x as one of a,b,c.
    # So we need a,b in S_old such that a+x in S_new, b+x in S_new, a+b in S_new.
    count = 0
    # Collect elements a in S_old where a+x is in S_new and a+x <= n
    A = []
    for a in S_old:
        ax = a + x
        if ax <= n and ax in s:
            A.append(a)
    for i in range(len(A)):
        a = A[i]
        for j in range(i+1, len(A)):
            b = A[j]
            ab = a + b
            if ab <= n and ab in s:
                count += 1
    return count

def _greedy_with_order(order, n):
    """Build a set by trying elements in given order, skipping if it creates a violation.
    Uses bitmask for fast membership checks.
    """
    S_mask = 0
    S_set = set()
    
    for x in order:
        if (S_mask >> x) & 1:
            continue
            
        # Tentatively add x
        S_mask_new = S_mask | (1 << x)
        
        # Efficient check:
        # If we add x, the new sums involving x are a+x.
        # A violation is a triple {a,b,x} where a,b in S_old.
        # Conditions:
        # 1. a+x in S_new
        # 2. b+x in S_new
        # 3. a+b in S_new
        
        valid = True
        A = []
        for a in range(1, n + 1):
            if (S_mask >> a) & 1: # a in S_old
                if a + x <= n and (S_mask_new >> (a + x)) & 1: # a+x in S_new
                    A.append(a)
        
        for i in range(len(A)):
            a = A[i]
            for j in range(i + 1, len(A)):
                b = A[j]
                if a + b <= n and (S_mask_new >> (a + b)) & 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S_mask = S_mask_new
            S_set.add(x)
            
    return S_set

def _simulated_annealing(rng, n, deadline, initial_set=None):
    """Simulated annealing to maximize set size while maintaining validity.
    Uses fast incremental violation checking."""
    if initial_set is None:
        initial_set = set(range(63, n + 1))
    
    current_set = set(initial_set)
    current_violations = _count_violations(current_set, n)
    
    best_set = set(current_set)
    best_violations = current_violations
    best_size = len(current_set)
    
    temperature = 1.0
    cooling_rate = 0.998
    min_temperature = 0.005
    
    # Precompute list for faster access
    all_elements = list(range(1, n + 1))
    
    while time.time() < deadline and temperature > min_temperature:
        # Choose a random move: add, remove, or swap
        move_type = rng.choice(['add', 'add', 'remove', 'swap', 'swap'])
        
        if move_type == 'add':
            candidates = [x for x in all_elements if x not in current_set]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_set = current_set | {x}
            # Fast incremental check
            new_violations = _violation_count_incremental(current_set, n) if False else None
            # Actually need to check if adding x creates violations
            # Use the incremental approach
            s_new = new_set
            count = 0
            A = []
            for a in current_set:
                ax = a + x
                if ax <= n and ax in s_new:
                    A.append(a)
            for i in range(len(A)):
                a = A[i]
                for j in range(i+1, len(A)):
                    b = A[j]
                    ab = a + b
                    if ab <= n and ab in s_new:
                        count += 1
            new_violations = current_violations + count
            new_size = len(new_set)
            
        elif move_type == 'remove':
            if not current_set:
                continue
            x = rng.choice(list(current_set))
            new_set = current_set - {x}
            # Removing can only reduce or maintain violations (actually it reduces)
            # Recompute is expensive; approximate: violations can only decrease
            # For simplicity, recompute if set is small enough, else estimate
            if len(new_set) <= 80:
                new_violations = _count_violations(new_set, n)
            else:
                # Estimate: removing x removes all violations involving x
                # This is an underestimate, but we'll recompute at the end
                new_violations = max(0, current_violations - 1)  # rough
            new_size = len(new_set)
        else:  # swap
            if len(current_set) < 2:
                continue
            in_list = list(current_set)
            out_list = [x for x in all_elements if x not in current_set]
            if not out_list:
                continue
            x_in = rng.choice(in_list)
            x_out = rng.choice(out_list)
            new_set = (current_set - {x_in}) | {x_out}
            # Recompute for swap (complex interaction)
            if len(new_set) <= 80:
                new_violations = _count_violations(new_set, n)
            else:
                new_violations = current_violations  # neutral estimate
            new_size = len(new_set)
        
        # Objective: minimize violations, maximize size
        delta_obj = 0
        if current_violations == 0 and new_violations == 0:
            delta_obj = -(new_size - len(current_set))
        elif current_violations == 0 and new_violations > 0:
            delta_obj = new_violations * 100
        elif current_violations > 0 and new_violations == 0:
            delta_obj = -current_violations * 100
        else:
            delta_obj = (new_violations - current_violations) * 10
            if delta_obj == 0:
                delta_obj = -(new_size - len(current_set))
        
        if delta_obj <= 0 or rng.random() < pow(2.718, -delta_obj / max(temperature, 0.001)):
            current_set = new_set
            # Recompute violations accurately when we accept
            if len(current_set) <= 85:
                current_violations = _count_violations(current_set, n)
            else:
                current_violations = new_violations
            
            if current_violations < best_violations or \
               (current_violations == best_violations and len(current_set) > best_size):
                best_set = set(current_set)
                best_violations = current_violations
                best_size = len(current_set)
        
        temperature *= cooling_rate
    
    # Final accurate check
    final_violations = _count_violations(best_set, n)
    if final_violations < best_violations or \
       (final_violations == best_violations and len(best_set) > best_size):
        best_violations = final_violations
        best_size = len(best_set)
    
    return best_set

def _polish_set(S, n=128, deadline=None):
    """Try to grow a valid set by single-element swaps. Returns improved set."""
    S = set(S)
    if not _is_valid(S):
        return S
    
    all_elements = list(range(1, n + 1))
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for x_out in all_elements:
            if x_out in S:
                continue
            for x_in in list(S):
                # Try swapping x_in with x_out
                new_set = (S - {x_in}) | {x_out}
                if _is_valid(new_set) and len(new_set) > len(S):
                    S = new_set
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    n = P['n']
    deadline = time.time() + 80.0  # safety margin before 90s limit
    
    # Baseline: {63, ..., 128} has size 66
    best_set = set(range(63, n + 1))
    best_size = len(best_set)
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy approaches (fast, good for initialization)
    for trial in range(300):
        if time.time() > deadline - 20.0:
            break
        
        # Random ordering biased toward upper half
        upper = list(range(63, n + 1))
        lower = list(range(1, 63))
        rng.shuffle(upper)
        rng.shuffle(lower)
        
        mix_ratio = rng.random()
        num_lower = int(len(lower) * mix_ratio)
        order = upper + lower[:num_lower]
        
        S = _greedy_with_order(order, n)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 2: Simulated Annealing
    sa_deadline = deadline - 15.0
    
    for trial in range(15):
        if time.time() > sa_deadline:
            break
        
        if trial == 0:
            start_set = set(best_set)
        elif trial == 1:
            # Start from upper half only
            start_set = set(range(60, n + 1))
        else:
            # Random start: mix of upper and lower
            upper = list(range(63, n + 1))
            lower = list(range(1, 63))
            rng.shuffle(upper)
            rng.shuffle(lower)
            num_upper = rng.randint(30, 60)
            num_lower = rng.randint(0, 20)
            start_set = set(upper[:num_upper] + lower[:num_lower])
        
        S = _simulated_annealing(rng, n, sa_deadline, initial_set=start_set)
        
        if _is_valid(S) and len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 3: Polish pass - try single swaps to grow the set
    polish_deadline = deadline - 5.0
    if time.time() < polish_deadline:
        polished = _polish_set(best_set, n, polish_deadline)
        if _is_valid(polished) and len(polished) > best_size:
            best_size = len(polished)
            best_set = polished
    
    # Final verification
    if _is_valid(best_set):
        return [int(x) for x in sorted(best_set)]
    else:
        # Fallback to known valid set
        return [int(x) for x in range(63, n + 1)]
# EVOLVE-BLOCK-END
