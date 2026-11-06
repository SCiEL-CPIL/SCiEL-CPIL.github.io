def construct():
    max_count = 0
    best_result = []
    
    # Try each prime P
    # Primes up to sqrt(128) approx 11, but we can check up to 128
    # Actually, we just need to check P such that floor(128/P) >= P
    # P^2 <= 128 => P <= 11
    # But let's just iterate all possible P
    for p in range(2, 129):
        # Check if p is prime (simple check)
        if p > 1:
            is_prime = True
            for i in range(2, int(p**0.5)+1):
                if p % i == 0:
                    is_prime = False
                    break
            if not is_prime:
                continue
                
        moduli = [p * k for k in range(1, 129) if p * k <= 128]
        count = len(moduli)
        
        # We can assign at most p distinct residues mod p
        # So max classes for this p is min(count, p)
        if count > 0:
            possible_classes = min(count, p)
            if possible_classes > max_count:
                max_count = possible_classes
                # Construct the result
                # We need 'possible_classes' moduli
                # Use the first 'possible_classes' multiples
                selected_moduli = moduli[:possible_classes]
                residues = list(range(possible_classes))
                res = [[r, m] for r, m in zip(residues, selected_moduli)]
                best_result = res

    return best_result
