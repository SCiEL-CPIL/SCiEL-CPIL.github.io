   def can_add(S, x):
       for d in range(1, N // 3 + 1):
           # Case 1: x is start
           if x + 3 * d <= N:
               if x + d in S and x + 2 * d in S and x + 3 * d in S:
                   return False
           
           # Case 2: x is 2nd
           if x - d >= 1 and x + 2 * d <= N:
               if x - d in S and x + d in S and x + 2 * d in S:
                   return False
           
           # Case 3: x is 3rd
           if x - 2 * d >= 1 and x + d <= N:
               if x - 2 * d in S and x - d in S and x + d in S:
                   return False
           
           # Case 4: x is 4th
           if x - 3 * d >= 1:
               if x - 3 * d in S and x - 2 * d in S and x - d in S:
                   return False
       
       return True
   