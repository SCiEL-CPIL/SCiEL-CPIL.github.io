import time
from fractions import Fraction

def construct():
       start_time = time.time()
       deadline = start_time + 140  # 10s margin

       # Generate primes up to 1000
       is_prime = [True] * 1001
       is_prime[0] = is_prime[1] = False
       for i in range(2, int(1000**0.5) + 1):
           if is_prime[i]:
               for j in range(i*i, 1001, i):
                   is_prime[j] = False
       primes = [i for i, flag in enumerate(is_prime) if flag]
       
       # State
       classes = [] # List of [p, r]
       used_primes = set()
       cost = Fraction(0, 1)
       
       # We will try to extend L
       L = 1
       covered = set()
       
       def update_coverage():
           # This is O(L * len(classes))
           # Too slow if called every step.
           # We should maintain covered incrementally.
           pass

       # Incremental coverage is hard to undo.
       # Let's just re-evaluate L from scratch periodically or use a faster check.
       
       # Alternative:
       # Just run the greedy until budget is tight or time runs out.
       
       # Greedy Step
       while True:
           if time.time() > deadline:
               break
               
           # Determine current max L covered
           # We maintain a boolean array `is_covered` up to some limit
           # But L can grow.
           
           # Let's assume we have a list of classes.
           # Find the first gap.
           n = 1
           while True:
               found = False
               for p, r in classes:
                   if n % p == r:
                       found = True
                       break
               if found:
                   n += 1
               else:
                   break
           L = n - 1
           
           if L >= 10000:
               break
           
           # Try to cover L+1
           target = L + 1
           
           best_p = None
           best_r = None
           best_score = -1
           
           for p in primes:
               if p in used_primes:
                   continue
               
               # Calculate coverage of target and future numbers?
               # Just check if it covers target.
               r = target % p
               
               # Score: Prefer small p (high density) but we need to fit budget.
               # If we can fit it, great.
               # If not, we might swap.
               
               # For now, just pick smallest p that covers target.
               # This minimizes cost per "direct" cover, but might not be optimal for global coverage.
               # However, small p covers many numbers.
               
               # Let's pick the smallest available p.
               if best_p is None or p < best_p:
                   best_p = p
                   best_r = r
                   
           if best_p is None:
               break
               
           new_cost = cost + Fraction(1, best_p)
           
           if new_cost <= 1:
               classes.append([best_p, best_r])
               used_primes.add(best_p)
               cost = new_cost
           else:
               # Try to swap
               # Remove largest prime in classes
               # Sort classes by p descending
               classes_sorted = sorted(classes, key=lambda x: x[0], reverse=True)
               swapped = False
               for p_old, r_old in classes_sorted:
                   # Check budget
                   if cost - Fraction(1, p_old) + Fraction(1, best_p) <= 1:
                       # Check if removing p_old breaks coverage of 1..L
                       # We need to verify that 1..L are still covered
                       # This is O(L * len(classes))
                       
                       # Temporarily remove
                       classes.remove([p_old, r_old])
                       used_primes.remove(p_old)
                       
                       # Check coverage
                       ok = True
                       for i in range(1, L + 1):
                           covered_i = False
                           for p_c, r_c in classes:
                               if i % p_c == r_c:
                                   covered_i = True
                                   break
                           if not covered_i:
                               ok = False
                               break
                           
                       if ok:
                           # Also check if target is covered by new class
                           if target % best_p == best_r:
                               classes.append([best_p, best_r])
                               used_primes.add(best_p)
                               cost = cost - Fraction(1, p_old) + Fraction(1, best_p)
                               swapped = True
                               break
                           else:
                               # Rollback
                               classes.append([p_old, r_old])
                               used_primes.add(p_old)
                       else:
                           # Rollback
                           classes.append([p_old, r_old])
                           used_primes.add(p_old)
               
               if not swapped:
                   break
                   
       return classes
