"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_set(bitset):
    """Score a bitset (int) representing a subset of 1..512.
    Returns number of distinct d > 0 such that a, a+d, a+2d all in set."""
    count = 0
    # d ranges from 1 to 255 (since 2d + 1 <= 512, d <= 255)
    for d in range(1, 256):
        # Check if there exists a such that a, a+d, a+2d are all set
        # bitset has bit i set if element (i+1) is in set (1-indexed)
        # a is at position a-1, a+d at a+d-1, a+2d at a+2d-1
        # We need: bitset has bits at positions p, p+d, p+2d for some p
        shifted1 = bitset >> d
        shifted2 = bitset >> (2 * d)
        if (bitset & shifted1 & shifted2) != 0:
            count += 1
    return count


def _make_bitset(lst):
    """Convert list of 1-indexed values to bitset int."""
    bs = 0
    for x in lst:
        bs |= (1 << (x - 1))
    return bs


def _bitset_to_list(bs):
    """Convert bitset to sorted list of 1-indexed values."""
    result = []
    for i in range(512):
        if (bs >> i) & 1:
            result.append(i + 1)
    return result


def _random_initial_set(n=32, size=512):
    """Generate a random initial set."""
    return random.sample(range(1, size + 1), n)


def _greedy_construct():
    """Greedy construction: build set to maximize distinct differences."""
    # Start with a spread-out set
    base = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # powers of 2 give many differences
    # Add more elements to fill up to 32
    remaining = [x for x in range(1, 513) if x not in base]
    random.shuffle(remaining)
    result = base + remaining[:32 - len(base)]
    return result


def _local_search_polish(lst, bitset, max_iters=20000, time_limit=None):
    """Local search: try swaps to improve score."""
    score = _score_set(bitset)
    n = len(lst)
    set_elems = set(lst)
    
    for it in range(max_iters):
        if time_limit and (time.time() - time_limit[0]) > time_limit[1]:
            break
        # Pick a random element to remove
        i = random.randint(0, n - 1)
        removed = lst[i]
        
        # Try adding a random new element
        new_elem = random.randint(1, 512)
        if new_elem in set_elems and new_elem != removed:
            continue
        
        # Build new bitset
        new_bitset = bitset & ~(1 << (removed - 1))
        new_bitset |= (1 << (new_elem - 1))
        
        new_score = _score_set(new_bitset)
        if new_score > score:
            score = new_score
            bitset = new_bitset
            lst[i] = new_elem
            set_elems.discard(removed)
            set_elems.add(new_elem)
    
    return lst, bitset, score


def _simulated_annealing(lst, bitset, score, start_temp=10.0, end_temp=0.1, 
                          max_iters=50000, time_limit=None):
    """Simulated annealing to find better sets."""
    n = len(lst)
    best_lst = lst[:]
    best_bitset = bitset
    best_score = score
    temp = start_temp
    cooling = (end_temp / start_temp) ** (1.0 / max(max_iters, 1))
    
    for it in range(max_iters):
        if time_limit and (time.time() - time_limit[0]) > time_limit[1]:
            break
        
        # Random move: swap one element
        i = random.randint(0, n - 1)
        old_elem = lst[i]
        new_elem = random.randint(1, 512)
        if new_elem == old_elem:
            continue
        
        # Check if new_elem is already in set
        if (bitset >> (new_elem - 1)) & 1:
            continue
        
        new_bitset = bitset & ~(1 << (old_elem - 1))
        new_bitset |= (1 << (new_elem - 1))
        new_score = _score_set(new_bitset)
        
        delta = new_score - score
        if delta > 0 or random.random() < np.exp(delta / temp):
            score = new_score
            bitset = new_bitset
            lst[i] = new_elem
            if score > best_score:
                best_score = score
                best_lst = lst[:]
                best_bitset = bitset
        
        temp *= cooling
    
    return best_lst, best_bitset, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Generate multiple initial sets
    random.seed(42)
    
    best_lst = None
    best_bitset = 0
    best_score = -1
    
    # Try several initial constructions
    initial_sets = []
    
    # Powers of 2 based
    initial_sets.append(_greedy_construct())
    
    # Random sets
    for _ in range(20):
        initial_sets.append(_random_initial_set(32, 512))
    
    # Consecutive blocks
    initial_sets.append(list(range(1, 33)))
    initial_sets.append(list(range(257, 289)))
    
    # Spread out
    initial_sets.append([1 + i * 16 for i in range(32)])
    initial_sets.append([1 + i * 15 for i in range(32)])
    
    # Evaluate all initial sets
    for lst in initial_sets:
        if len(set(lst)) != 32:
            continue
        bitset = _make_bitset(lst)
        score = _score_set(bitset)
        if score > best_score:
            best_score = score
            best_lst = lst[:]
            best_bitset = bitset
    
    # Run local search polish on best
    best_lst, best_bitset, best_score = _local_search_polish(
        best_lst, best_bitset, max_iters=10000, time_limit=[start_time, deadline]
    )
    
    # Run simulated annealing
    if time.time() < deadline - 5:
        best_lst, best_bitset, best_score = _simulated_annealing(
            best_lst, best_bitset, best_score,
            start_temp=5.0, end_temp=0.05,
            max_iters=30000, time_limit=[start_time, deadline]
        )
        
        # Final polish
        if time.time() < deadline - 2:
            best_lst, best_bitset, best_score = _local_search_polish(
                best_lst, best_bitset, max_iters=5000, time_limit=[start_time, deadline]
            )
    
    # Validate
    result = sorted(best_lst)
    if len(result) != 32 or len(set(result)) != 32 or any(x < 1 or x > 512 for x in result):
        # Fallback
        result = list(range(1, 33))
    
    return result

# EVOLVE-BLOCK-END
