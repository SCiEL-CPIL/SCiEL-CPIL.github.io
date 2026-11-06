import time

def construct():
       deadline = time.time() + 85.0
       N = 123457
       
       # Generate primes up to sqrt(N)
       limit = int(N**0.5) + 1
       is_prime = [True] * (limit + 1)
       is_prime[0] = is_prime[1] = False
       for i in range(2, int(limit**0.5) + 1):
           if is_prime[i]:
               for j in range(i*i, limit + 1, i):
                   is_prime[j] = False
       primes = [i for i, v in enumerate(is_prime) if v]
       
       LPF = [1] * (N + 1)
       
       for i in range(2, N + 1):
           if time.time() > deadline - 2.0:
               # If we run out of time, we might have partial LPF.
               # But this loop is fast.
               break
           n = i
           max_p = 1
           for p in primes:
               if p * p > n:
                   break
               if n % p == 0:
                   max_p = p
                   while n % p == 0:
                       n //= p
           if n > 1:
               max_p = n
           LPF[i] = max_p
           
       best_val = float('inf')
       best_a = 1
       for a in range(1, N):
           val = max(LPF[a], LPF[N - a])
           if val < best_val:
               best_val = val
               best_a = a
               if best_val == 2: # Theoretical min for even N? No, N is odd.
                   # If N is odd, one is even (div by 2), one is odd.
                   # Min possible max is at least 2?
                   # If a=2, P(2)=2. b=N-2 (odd). P(b) >= 3.
                   # So min is at least 3?
                   # Actually, if a is power of 2, P(a)=2.
                   # b = N - 2^k.
                   # We want P(b) small.
                   pass
                   
       return [best_a, N - best_a]
