import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 1000000
    # We need P(n) for n up to N, so we compute up to N+1 to get P(N+1) for n=N? 
    # Wait, n is in [10000, 1000000]. The max product involves P(n) and P(n+1).
    # If n=1000000, we need P(1000001).
    # So we need to compute P up to 1000001.
    MAX_VAL = N + 1
    
    # Initialize powerful part array
    # pp[i] will store the powerful part of i
    pp = [1] * (MAX_VAL + 1)
    
    # Sieve for primes and powers
    # We need to identify primes. A simple sieve is sufficient.
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_VAL + 1, i):
                is_prime[j] = False
                
    # For each prime, update the powerful parts of its multiples
    for p in range(2, MAX_VAL + 1):
        if is_prime[p]:
            # Calculate powers p^2, p^3, ...
            power = p * p
            while power <= MAX_VAL:
                # Multiply pp[m] by power for all m divisible by power
                for m in range(power, MAX_VAL + 1, power):
                    pp[m] *= power
                power *= p
                
    # Find the best n
    best_n = 10000
    best_score = 0
    
    # n must be in [10000, 1000000]
    # We check P(n) * P(n+1)
    for n in range(10000, N + 1):
        score = pp[n] * pp[n+1]
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-START
# The function construct is defined above, but the prompt asks to keep the function names 
# and signatures of the seed program. The seed program had construct inside the evolve block?
# No, the seed program had:
# def construct():
#     return 10000
# inside the block.
# However, usually helper functions are outside or inside. 
# The instruction says: "Keep the function names and signatures of the seed program."
# The seed program only had `construct`.
# I will place the helper logic inside `construct` or as module level functions if needed, 
# but to be safe and self-contained within the evolve block structure if required, 
# I'll define the logic directly in construct or use module-level helpers if the 
# "EVOLVE-BLOCK" implies only that section is replaced.
# The prompt says: "Rewrite the program from scratch... Output the COMPLETE program".
# So I can define helper functions outside the block if I want, but the block markers 
# in the example suggest the content between them is the evolving part.
# I will put everything inside the EVOLVE-BLOCK to be safe, or just define construct 
# properly. The seed had construct inside the block.
# I will define construct inside the block as requested by the structure.
# EVOLVE-BLOCK-END
