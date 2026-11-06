# EVOLVE-BLOCK-START
def construct():
    # Find all primes up to 128
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [i for i in range(2, 129) if is_prime(i)]
    
    # Assign residue 0 to all primes
    # They are pairwise coprime, so any residues work
    result = [[0, p] for p in primes]
    
    return result
# EVOLVE-BLOCK-END
