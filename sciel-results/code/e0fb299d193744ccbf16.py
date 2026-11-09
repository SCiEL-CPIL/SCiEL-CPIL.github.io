   S = []
   pairs_in_S = set()
   for idx in order:
       if _can_add(idx, pairs_in_S):
           S.append(idx)
           for p in S[:-1]:
               a, b = idx, p
               if a > b: a, b = b, a
               pairs_in_S.add((a, b))
   