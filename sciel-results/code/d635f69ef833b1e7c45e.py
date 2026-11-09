   A.add(y)
   in_A[y] = True
   for z in A: # A now contains y
       s = y + z
       if target_min <= s <= target_max:
           counts[s] += 1
   