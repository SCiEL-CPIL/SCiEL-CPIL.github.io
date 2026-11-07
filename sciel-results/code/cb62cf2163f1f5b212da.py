   for p in primes:
       val = p
       while val * p <= n:
           val *= p
       result.append(val)
   