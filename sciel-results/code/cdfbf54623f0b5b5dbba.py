import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    def is_valid_sequence(seq):
        if not seq:
            return True
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def local_search(seq):
        improved = True
        seq_set = set(seq)
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            
            # Try Insertions
            # We want to try inserting numbers that are "likely" to fit.
            # Smaller numbers fit earlier, larger later.
            # Just iterate all missing numbers.
            for x in range(2, 129):
                if x in seq_set:
                    continue
                # Find position
                pos = 0
                while pos < len(seq) and seq[pos] < x:
                    pos += 1
                new_seq = seq[:pos] + [x] + seq[pos:]
                if is_valid_sequence(new_seq):
                    seq = new_seq
                    seq_set = set(seq)
                    improved = True
                    break # Restart to re-evaluate all insertions
            
            # Try Replacements if no insertion found
            if not improved:
                # Try to replace an element with another
                # Randomize the order of indices to avoid bias
                indices = list(range(len(seq)))
                random.shuffle(indices)
                for idx in indices:
                    x = seq[idx]
                    # Try a few random replacements
                    for _ in range(20):
                        y = random.randint(2, 128)
                        if y == x or y in seq_set:
                            continue
                        # Remove x, insert y
                        new_seq = seq[:idx] + seq[idx+1:]
                        # Insert y
                        pos = 0
                        while pos < len(new_seq) and new_seq[pos] < y:
                            pos += 1
                        new_seq = new_seq[:pos] + [y] + new_seq[pos:]
                        if is_valid_sequence(new_seq):
                            seq = new_seq
                            seq_set = set(seq)
                            improved = True
                            break
                    if improved:
                        break

        return seq

    best_seq = []
    
    # Initial Greedy
    seq = []
    for x in range(2, 129):
        candidate = seq + [x]
        if is_valid_sequence(candidate):
            seq = candidate
    best_seq = local_search(seq)
    
    # Restart Loop
    while time.time() - start_time < DEADLINE:
        # Randomized Greedy
        seq = []
        for x in range(2, 129):
            candidate = seq + [x]
            if is_valid_sequence(candidate):
                # With some probability, skip to diversify
                # If we skip, we might allow a different structure
                # But we want max length, so skipping is risky.
                # Let's use a lower skip prob, e.g., 0.1
                if random.random() > 0.1:
                    seq = candidate
        
        improved_seq = local_search(seq)
        if len(improved_seq) > len(best_seq):
            best_seq = improved_seq

    return best_seq

if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Length: {len(res)}")
    print(f"Time: {end - start:.2f}s")
    
    def verify(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True
    print("Valid:", verify(res))
