import random
import math
from itertools import combinations

def construct():
       # Seed solution
       best_A = [1, 7, 13]
       best_B = [4, 10, 16]
       best_score = 5
       
       # Precompute primes
       is_prime = [True] * 257
       is_prime[0] = is_prime[1] = False
       for i in range(2, int(256**0.5) + 1):
           if is_prime[i]:
               for j in range(i*i, 257, i):
                   is_prime[j] = False
               
       def score(A, B):
           sums = []
           for a in A:
               for b in B:
                   sums.append(a + b)
           
           # Check distinct
           if len(set(sums)) < 9:
               distinct_sums = list(set(sums))
               n = len(distinct_sums)
               if n < best_score:
                   return 0
               for i in range(n):
                   for j in range(i+1, n):
                       if math.gcd(distinct_sums[i], distinct_sums[j]) != 1:
                           return 0
               return n
           
           # If 9 distinct, check pairwise coprime
           for i in range(9):
               for j in range(i+1, 9):
                   if math.gcd(sums[i], sums[j]) != 1:
                       return 0
           return 9

       # Check seed
       s = score(best_A, best_B)
       if s > best_score:
           best_score = s
           
       # Candidate B sets: small numbers to facilitate small sums which are easier to be primes
       # We want sums to be distinct and coprime. Primes are coprime to each other (except 2).
       # If we can make all 9 sums distinct primes, we get score 9.
       
       candidate_Bs = [
           [1, 2, 3],
           [1, 2, 4],
           [1, 2, 5],
           [1, 3, 4],
           [1, 3, 5],
           [1, 3, 6],
           [1, 4, 5],
           [1, 4, 7],
           [1, 5, 7],
           [2, 3, 5],
           [2, 3, 7],
           [2, 5, 7],
           [3, 4, 5],
           [3, 5, 7],
           [4, 5, 7],
           [1, 2, 8],
           [1, 2, 9],
           [1, 3, 8],
           [1, 3, 9],
           [1, 4, 8],
           [1, 4, 9],
           [1, 5, 8],
           [1, 5, 9],
           [1, 6, 7],
           [1, 6, 8],
           [1, 6, 9],
           [1, 7, 8],
           [1, 7, 9],
           [1, 8, 9],
           [2, 3, 4],
           [2, 3, 6],
           [2, 3, 8],
           [2, 3, 9],
           [2, 4, 5],
           [2, 4, 7],
           [2, 4, 9],
           [2, 5, 9],
           [2, 6, 7],
           [2, 6, 9],
           [2, 7, 9],
           [3, 4, 7],
           [3, 4, 8],
           [3, 4, 9],
           [3, 5, 8],
           [3, 5, 9],
           [3, 6, 7],
           [3, 6, 8],
           [3, 6, 9],
           [3, 7, 8],
           [3, 7, 9],
           [3, 8, 9],
           [4, 5, 9],
           [4, 7, 9],
           [4, 8, 9],
           [5, 7, 9],
           [5, 8, 9],
           [7, 8, 9]
       ]
       
       # Shuffle B sets to avoid bias
       random.shuffle(candidate_Bs)
       
       # We will try each B set with many random A sets
       # If we find a score of 9, we return immediately
       
       for B in candidate_Bs:
           # Try 20000 random A sets for this B
           for _ in range(20000):
               A = list(random.sample(range(1, 129), 3))
               s = score(A, B)
               if s > best_score:
                   best_score = s
                   best_A = A
                   best_B = B
                   if best_score == 9:
                       return [best_A, best_B]
                   
       # If we didn't find 9, try a more exhaustive search on the best B found so far?
       # Or just return the best we have.
       
       return [best_A, best_B]
