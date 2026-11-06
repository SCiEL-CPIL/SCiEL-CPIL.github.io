"""Stronger algorithm for dense increasing sequences with unique block products."""
import time
import random
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _validate(seq):
    """Check if seq is strictly increasing in [2,128] and all block products distinct."""
    if not seq:
        return True
    if seq[0] < 2 or seq[-1] > 128:
        return False
    for i in range(1, len(seq)):
        if seq[i] <= seq[i - 1]:
            return False
    n = len(seq)
    prods = set()
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return len(prods) == n * (n + 1) // 2


def _compute_products(seq):
    """Compute all block products for a sequence."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods


def _check_insertion(seq, pos, val, existing_prods):
    """
    Check if inserting val at position pos in seq keeps all block products distinct.
    Only checks new products that involve val.
    Returns (is_valid, new_prods_set) if valid, else (False, None).
    """
    n = len(seq)
    new_prods = set()
    
    # Products that start at pos (val * seq[pos] * ... * seq[pos+k])
    p = val
    for k in range(n - pos):
        new_prods.add(p)
        if k + 1 < n - pos:
            p *= seq[pos + k + 1]
    
    # Products that end at pos (seq[pos-k] * ... * seq[pos-1] * val)
    p = val
    for k in range(pos):
        new_prods.add(p)
        p *= seq[pos - k - 1]
    
    # Check against existing products
    if new_prods & existing_prods:
        return False, None
    
    return True, new_prods


def _greedy_with_order(order):
    """Greedy: try each x in given order, add if all new tail products are fresh."""
    values, products = [], set()
    for x in order:
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _primes_first_greedy():
    """Start with all primes, then greedily insert composites."""
    primes = [x for x in range(2, 129) if _is_prime(x)]
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    
    seq = list(primes)
    prods = _compute_products(seq)
    
    for c in composites:
        for pos in range(len(seq) + 1):
            if pos > 0 and seq[pos - 1] >= c:
                continue
            if pos < len(seq) and seq[pos] <= c:
                continue
            
            valid, new_prods = _check_insertion(seq, pos, c, prods)
            if valid:
                seq = seq[:pos] + [c] + seq[pos:]
                prods = set(prods) | new_prods
                break
    
    return seq


def _beam_search(deadline, beam_width=10, max_depth=40):
    """
    Beam search over sequences. Each beam state is (seq, prods_set).
    At each step, try inserting each unused number at each valid position,
    keep the top beam_width candidates by length (then by some heuristic).
    """
    primes = [x for x in range(2, 129) if _is_prime(x)]
    
    # Initialize beam with just primes
    seq = list(primes)
    prods = _compute_products(seq)
    beam = [(seq, prods)]
    
    all_nums = set(range(2, 129))
    
    for depth in range(max_depth):
        if time.time() > deadline - 5:
            break
        
        new_beam = []
        seen_seqs = set()
        
        for seq, prods in beam:
            used = set(seq)
            unused = [x for x in all_nums if x not in used]
            
            # Try each unused number
            for c in unused:
                # Find valid positions
                for pos in range(len(seq) + 1):
                    if pos > 0 and seq[pos - 1] >= c:
                        continue
                    if pos < len(seq) and seq[pos] <= c:
                        continue
                    
                    valid, new_prods = _check_insertion(seq, pos, c, prods)
                    if valid:
                        new_seq = tuple(seq[:pos] + [c] + seq[pos:])
                        if new_seq not in seen_seqs:
                            seen_seqs.add(new_seq)
                            new_beam.append((list(new_seq), set(prods) | new_prods))
            
            if time.time() > deadline - 5:
                break
        
        if not new_beam:
            break
        
        # Sort by length descending, then by sum of elements (prefer denser)
        new_beam.sort(key=lambda x: (-len(x[0]), -sum(x[0])))
        beam = new_beam[:beam_width]
    
    # Return the best from the final beam
    if beam:
        return max(beam, key=lambda x: len(x[0]))[0]
    return seq


def _randomized_hill_climbing(deadline, num_restarts=50):
    """
    Multiple random restarts with hill climbing.
    Each restart: start from a random valid sequence, then greedily insert.
    """
    best_seq = []
    best_len = 0
    all_nums = set(range(2, 129))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        # Start with a random subset of primes
        primes = [x for x in range(2, 129) if _is_prime(x)]
        rng.shuffle(primes)
        seq = primes[:len(primes)//2 + rng.randint(0, 5)]
        seq.sort()
        
        # Verify initial sequence
        prods = _compute_products(seq)
        if len(prods) != len(seq) * (len(seq) + 1) // 2:
            continue
        
        # Greedily insert remaining numbers
        used = set(seq)
        unused = [x for x in all_nums if x not in used]
        rng.shuffle(unused)
        
        for c in unused:
            if time.time() > deadline - 5:
                break
            for pos in range(len(seq) + 1):
                if pos > 0 and seq[pos - 1] >= c:
                    continue
                if pos < len(seq) and seq[pos] <= c:
                    continue
                
                valid, new_prods = _check_insertion(seq, pos, c, prods)
                if valid:
                    seq = seq[:pos] + [c] + seq[pos:]
                    prods = set(prods) | new_prods
                    break
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
    
    return best_seq


def _swap_local_search(seq, deadline, max_iterations=500):
    """
    Local search: try inserting unused elements, and if stuck, try swapping
    adjacent elements to unlock new insertions.
    """
    seq = list(seq)
    prods = _compute_products(seq)
    best = list(seq)
    best_len = len(seq)
    all_nums = set(range(2, 129))
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        used = set(seq)
        unused = [x for x in all_nums if x not in used]
        random.shuffle(unused)
        
        # Try inserting unused numbers
        for c in unused:
            if time.time() > deadline:
                break
            for pos in range(len(seq) + 1):
                if pos > 0 and seq[pos - 1] >= c:
                    continue
                if pos < len(seq) and seq[pos] <= c:
                    continue
                
                valid, new_prods = _check_insertion(seq, pos, c, prods)
                if valid:
                    seq = seq[:pos] + [c] + seq[pos:]
                    prods = set(prods) | new_prods
                    if len(seq) > best_len:
                        best = list(seq)
                        best_len = len(seq)
                    improved = True
                    break
            if improved:
                break
        
        # If no insertion, try swapping adjacent elements
        if not improved:
            for i in range(len(seq) - 1):
                if time.time() > deadline:
                    break
                
                seq[i], seq[i+1] = seq[i+1], seq[i]
                new_prods = _compute_products(seq)
                
                if len(new_prods) == len(seq) * (len(seq) + 1) // 2:
                    # Valid swap, try inserting
                    used = set(seq)
                    unused = [x for x in all_nums if x not in used]
                    random.shuffle(unused)
                    
                    for c in unused:
                        if time.time() > deadline:
                            break
                        for pos in range(len(seq) + 1):
                            if pos > 0 and seq[pos - 1] >= c:
                                continue
                            if pos < len(seq) and seq[pos] <= c:
                                continue
                            
                            valid, new_prods_insert = _check_insertion(seq, pos, c, new_prods)
                            if valid:
                                seq = seq[:pos] + [c] + seq[pos:]
                                prods = set(new_prods) | new_prods_insert
                                if len(seq) > best_len:
                                    best = list(seq)
                                    best_len = len(seq)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                else:
                    seq[i], seq[i+1] = seq[i+1], seq[i]
        
        if not improved:
            break
    
    return best


def _simulated_annealing(deadline, start_temp=10.0, cool_rate=0.95):
    """
    Simulated annealing: accept worse solutions with probability exp(-delta/T).
    Moves: insert a random unused element at a random valid position, or swap adjacent.
    """
    primes = [x for x in range(2, 129) if _is_prime(x)]
    seq = list(primes)
    prods = _compute_products(seq)
    best = list(seq)
    best_len = len(seq)
    all_nums = set(range(2, 129))
    
    temp = start_temp
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        temp *= cool_rate
        if temp < 0.01:
            break
        
        used = set(seq)
        unused = [x for x in all_nums if x not in used]
        
        if not unused:
            break
        
        # Random move
        move_type = rng.random()
        
        if move_type < 0.7 and unused:
            # Try inserting a random unused element
            c = rng.choice(unused)
            valid_positions = []
            for pos in range(len(seq) + 1):
                if pos > 0 and seq[pos - 1] >= c:
                    continue
                if pos < len(seq) and seq[pos] <= c:
                    continue
                valid_positions.append(pos)
            
            if valid_positions:
                pos = rng.choice(valid_positions)
                valid, new_prods = _check_insertion(seq, pos, c, prods)
                if valid:
                    new_seq = seq[:pos] + [c] + seq[pos:]
                    new_prods_full = set(prods) | new_prods
                    delta = len(new_seq) - len(seq)
                    
                    if delta >= 0 or rng.random() < pow(2.718281828, -delta / max(temp, 0.01)):
                        seq = new_seq
                        prods = new_prods_full
                        if len(seq) > best_len:
                            best = list(seq)
                            best_len = len(seq)
        else:
            # Try swapping adjacent elements
            if len(seq) > 1:
                i = rng.randint(0, len(seq) - 2)
                seq[i], seq[i+1] = seq[i+1], seq[i]
                new_prods = _compute_products(seq)
                
                if len(new_prods) == len(seq) * (len(seq) + 1) // 2:
                    # Valid, maybe try inserting after swap
                    used = set(seq)
                    unused = [x for x in all_nums if x not in used]
                    if unused:
                        c = rng.choice(unused)
                        valid_positions = []
                        for pos in range(len(seq) + 1):
                            if pos > 0 and seq[pos - 1] >= c:
                                continue
                            if pos < len(seq) and seq[pos] <= c:
                                continue
                            valid_positions.append(pos)
                        
                        if valid_positions:
                            pos = rng.choice(valid_positions)
                            valid, new_prods_insert = _check_insertion(seq, pos, c, new_prods)
                            if valid:
                                new_seq = seq[:pos] + [c] + seq[pos:]
                                new_prods_full = set(new_prods) | new_prods_insert
                                delta = len(new_seq) - len(seq)
                                
                                if delta >= 0 or rng.random() < pow(2.718281828, -delta / max(temp, 0.01)):
                                    seq = new_seq
                                    prods = new_prods_full
                                    if len(seq) > best_len:
                                        best = list(seq)
                                        best_len = len(seq)
                else:
                    seq[i], seq[i+1] = seq[i+1], seq[i]
    
    return best


def construct():
    deadline = time.time() + 80.0  # safety margin before 90s limit
    candidates = []
    
    # Strategy 1: Primes-first greedy (strong baseline)
    g1 = _primes_first_greedy()
    candidates.append(g1)
    
    # Strategy 2: Beam search
    if time.time() < deadline - 20:
        bs = _beam_search(deadline - 20, beam_width=15, max_depth=30)
        candidates.append(bs)
    
    # Strategy 3: Randomized hill climbing
    if time.time() < deadline - 15:
        rhc = _randomized_hill_climbing(deadline - 15, num_restarts=30)
        candidates.append(rhc)
    
    # Strategy 4: Simple greedy variants
    g2 = _greedy_with_order(range(2, 129))
    candidates.append(g2)
    
    primes = [x for x in range(2, 129) if _is_prime(x)]
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    g3 = _greedy_with_order(primes + composites)
    candidates.append(g3)
    
    # Strategy 5: Simulated annealing
    if time.time() < deadline - 10:
        sa = _simulated_annealing(deadline - 10)
        candidates.append(sa)
    
    # Strategy 6: Local search from best candidate
    best_start = max(candidates, key=len)
    if time.time() < deadline - 5:
        ls_result = _swap_local_search(best_start, deadline, max_iterations=300)
        candidates.append(ls_result)
    
    # Pick the longest valid result
    best = max(candidates, key=len)
    
    # Final validation
    if not _validate(best):
        best = _primes_first_greedy()
        if not _validate(best):
            best = _greedy_with_order(range(2, 129))
    
    return best

# EVOLVE-BLOCK-END
