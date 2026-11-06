import random
import time

def _check_valid(seq):
    """Check if seq is strictly increasing in [2,128] and all consecutive block products are distinct."""
    if not seq:
        return False
    n = len(seq)
    # Check strictly increasing and range
    for i in range(n):
        if seq[i] < 2 or seq[i] > 128:
            return False
        if i > 0 and seq[i] <= seq[i-1]:
            return False
    # Check all consecutive block products are distinct
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def _greedy_init():
    """Build an initial valid sequence using greedy method."""
    values = []
    products = set()
    for x in range(2, 129):
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def construct():
    """
    Use simulated annealing to find a long valid sequence.
    Returns strictly increasing integers in [2,128] with distinct consecutive block products.
    """
    start_time = time.time()
    time_limit = 55.0  # leave 5s buffer

    # Initialize with greedy solution
    best_seq = _greedy_init()
    best_len = len(best_seq)

    # Simulated annealing
    current_seq = list(best_seq)
    current_len = len(current_seq)

    # Temperature parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.9995

    T = T_start
    iterations = 0
    max_iterations = 200000

    while T > T_end and iterations < max_iterations:
        if time.time() - start_time > time_limit:
            break
        iterations += 1

        # Generate a random move
        move_type = random.random()
        new_seq = list(current_seq)

        if move_type < 0.4:
            # Insert a random number at a random position
            candidates = [x for x in range(2, 129) if x not in new_seq]
            if candidates:
                x = random.choice(candidates)
                pos = random.randint(0, len(new_seq))
                new_seq.insert(pos, x)
        elif move_type < 0.7:
            # Remove a random number
            if len(new_seq) > 1:
                pos = random.randint(0, len(new_seq) - 1)
                new_seq.pop(pos)
        else:
            # Swap two random elements (then sort to maintain increasing order)
            if len(new_seq) >= 2:
                i, j = random.sample(range(len(new_seq)), 2)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                new_seq.sort()  # maintain strictly increasing
                # After sort, check for duplicates (shouldn't happen if all unique)

        # Validate new sequence
        if _check_valid(new_seq):
            new_len = len(new_seq)
            # Accept if better or with probability exp((new_len - current_len) / T)
            if new_len >= current_len:
                current_seq = new_seq
                current_len = new_len
                if current_len > best_len:
                    best_seq = list(current_seq)
                    best_len = current_len
            else:
                # Worse: accept with probability
                delta = new_len - current_len  # negative
                prob = min(1.0, 2.718281828 ** (delta / T))
                if random.random() < prob:
                    current_seq = new_seq
                    current_len = new_len
        else:
            # Invalid: reject
            pass

        # Cool down
        T *= cooling_rate

    return best_seq
