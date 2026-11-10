def construct():
       def count_divisors(n):
           count = 0
           i = 1
           while i * i <= n:
               if n % i == 0:
                   count += 1
                   if i != n // i:
                       count += 1
               i += 1
           return count

       best_m = 1
       max_count = 0
       for m in range(1, 513):
           c = count_divisors(m)
           if c > max_count:
               max_count = c
               best_m = m

       # Return divisors of best_m
       divs = []
       for i in range(1, int(best_m**0.5) + 1):
           if best_m % i == 0:
               divs.append(i)
               if i != best_m // i:
                   divs.append(best_m // i)
       return divs
