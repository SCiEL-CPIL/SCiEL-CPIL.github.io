import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    def is_valid(seq):
        """Check if all consecutive block products are distinct."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def greedy_construct():
        """Greedy construction: add numbers in order of increasing value."""
        seq = []
        for x in range(2, 129):
            if time.time() > deadline:
                break
            # Find insertion position
            pos = 0
            while pos < len(seq) and seq[pos] < x:
                pos += 1
            
            # Check validity with insertion
            new_seq = seq[:pos] + [x] + seq[pos:]
            if is_valid(new_seq):
                seq = new_seq
        return seq

    def local_search(seq):
        """
        Local search to maximize length.
        1. Try to insert any unused number.
        2. If no insertion, try a random valid swap (remove one, add one) to change structure.
        """
        best_seq = seq[:]
        best_len = len(seq)
        used = set(best_seq)
        
        while time.time() < deadline:
            improved = False
            
            # Step 1: Try to insert any unused number
            for x in range(2, 129):
                if x in used:
                    continue
                pos = 0
                while pos < len(best_seq) and best_seq[pos] < x:
                    pos += 1
                
                new_seq = best_seq[:pos] + [x] + best_seq[pos:]
                if is_valid(new_seq):
                    best_seq = new_seq
                    used.add(x)
                    best_len = len(best_seq)
                    improved = True
                    break # Restart insertion search with new state
            
            if improved:
                continue
            
            # Step 2: If no insertion, try a random swap to escape local minimum
            # Pick a random element to remove
            if len(best_seq) == 0:
                break
                
            # To be efficient, we might not want to try ALL swaps. 
            # Let's try a few random swaps.
            found_swap = False
            # Limit swaps per iteration to avoid getting stuck in one loop
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                # Randomly pick x to remove
                x = random.choice(best_seq)
                # Randomly pick y to add (not in best_seq)
                available = [y for y in range(2, 129) if y not in used]
                if not available:
                    break
                y = random.choice(available)
                
                # Form new sequence
                new_seq = [v for v in best_seq if v != x]
                # Insert y
                pos = 0
                while pos < len(new_seq) and new_seq[pos] < y:
                    pos += 1
                new_seq = new_seq[:pos] + [y] + new_seq[pos:]
                
                if is_valid(new_seq):
                    # Accept the swap (length unchanged, structure changed)
                    best_seq = new_seq
                    used.discard(x)
                    used.add(y)
                    found_swap = True
                    break
            
            if not found_swap:
                # If we can't find a swap, we are stuck. 
                # Maybe try a 2-swap? Or just break?
                # Given the density, 1-swap should usually work if a better config exists nearby.
                # If stuck, we might need to remove an element to allow a different path, 
                # but that reduces length. We only accept length-increasing or length-preserving moves.
                # If we can't preserve length, we are at a local optimum for this branch.
                # We could break and rely on restarts.
                break
                
        return best_seq

    # Initial greedy solution
    best_seq = greedy_construct()
    
    # Local search
    best_seq = local_search(best_seq)
    
    # Try a few random restarts if time permits
    rng = random.Random(42)
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        
        # Random greedy
        candidates = list(range(2, 129))
        rng.shuffle(candidates)
        seq = []
        for x in candidates:
            if time.time() > deadline - 2:
                break
            pos = 0
            while pos < len(seq) and seq[pos] < x:
                pos += 1
            new_seq = seq[:pos] + [x] + seq[pos:]
            if is_valid(new_seq):
                seq = new_seq
        
        # Local search on this random start
        seq = local_search(seq)
        
        if len(seq) > len(best_seq):
            best_seq = seq
            
    return best_seq
