import math
import time

def construct():
       start_time = time.time()
       deadline = start_time + 85.0 # 5s margin

       n = 210
       max_val = 10210
       
       # Sieve of Eratosthenes for primes up to max_val
       is_prime = [True] * (max_val + 1)
       is_prime[0] = is_prime[1] = False
       for i in range(2, int(max_val**0.5) + 1):
           if is_prime[i]:
               for j in range(i*i, max_val + 1, i):
                   is_prime[j] = False
               
       primes = [i for i, val in enumerate(is_prime) if val]
       prime_index = {p: i for i, p in enumerate(primes)}
       num_primes = len(primes)

       def get_mask(x):
           mask = 0
           temp = x
           for p in primes:
               if p * p > temp:
                   break
               if temp % p == 0:
                   count = 0
                   while temp % p == 0:
                       temp //= p
                       count ^= 1
                   if count:
                       mask |= (1 << prime_index[p])
           if temp > 1:
               # temp is a prime > sqrt(x)
               # It must be in our primes list if x <= max_val
               # Because if temp is a factor of x, and x <= max_val, temp <= max_val
               if temp in prime_index:
                   mask |= (1 << prime_index[temp])
           return mask

       target_mask = get_mask(n)
       
       # Basis: list of (vector, history_mask)
       # We use a dict for O(1) lookup by highest bit
       basis = {}
       
       # We will store candidate values in a list to map history bits to values
       # History bit i corresponds to candidate at index i
       # Candidate i is value 211 + i
       
       for M in range(211, max_val + 1):
           if time.time() > deadline:
               # Fallback: return a known valid set if we have one, or just break
               # But we should have found one by now
               break
               
           v = get_mask(M)
           h = 1 << (M - 211) # Index of this candidate in the sequence 211..M
           
           # Insert into basis
           temp_v = v
           temp_h = h
           while temp_v:
               # Find highest set bit
               # Python int has no direct 'highest bit' index, but we can use bit_length
               p = temp_v.bit_length() - 1
               if p in basis:
                   b_vec, b_hist = basis[p]
                   temp_v ^= b_vec
                   temp_h ^= b_hist
               else:
                   basis[p] = (temp_v, temp_h)
                   break
           
           # Check if target is in span
           # Reduce target using basis
           t_v = target_mask
           t_h = 0
           while t_v:
               p = t_v.bit_length() - 1
               if p in basis:
                   b_vec, b_hist = basis[p]
                   t_v ^= b_vec
                   t_h ^= b_hist
               else:
                   break
           
           if t_v == 0:
               # Solution found
               # t_h is a bitmask of indices
               # Indices correspond to values 211 + i
               sol_indices = []
               # Iterate through bits set in t_h
               # t_h can be large, so iterate efficiently
               # We only care about indices < M - 210
               hh = t_h
               while hh:
                   # Get lowest set bit
                   lsb = hh & -hh
                   idx = lsb.bit_length() - 1
                   sol_indices.append(211 + idx)
                   hh ^= lsb
               
               sol = sol_indices
               # Verify size
               if len(sol) <= 100:
                   # Verify product is square
                   prod = n
                   for x in sol:
                       prod *= x
                   if math.isqrt(prod)**2 == prod:
                       return [int(x) for x in sol]
               else:
                   # If size > 100, we might need to continue searching for a better M
                   # or try to minimize this solution
                   # For now, continue to see if a smaller M (which we already passed) or 
                   # a different combination exists? 
                   # Actually, M is increasing. The first M that works is the minimal M.
                   # But the solution found might not be the smallest subset for that M.
                   # However, if M is small, len(sol) is small.
                   # If M is large, len(sol) might be large.
                   # If we reach here, it means the first valid M yielded a large subset.
                   # We can try to find a smaller subset for this M by solving the system differently?
                   # Or just accept it if we can't do better quickly.
                   # But the task says minimize max(selected). So M is the primary objective.
                   # If the only solution for minimal M has size > 100, it's invalid.
                   # We must find a valid solution.
                   # So we should continue to next M? No, M is the max value.
                   # If M is the max value in the set, then all elements are <= M.
                   # If we go to M+1, we can still use elements <= M.
                   # But the max will be at least M.
                   # So if the solution for M is invalid (size > 100), we must find another solution for M 
                   # or a solution for M+1 with max <= M+1.
                   # Actually, if we add M+1, we can form a new subset that might be smaller.
                   # So we should continue the loop.
                   pass

       # If loop finishes without returning, return a fallback
       # This should not happen
       return [4*n]
