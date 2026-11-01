   masks_local = masks
   for i in range(len(available)):
       idx = available[i]
       m = masks_local[idx]
       # ...
       new_avail = []
       for j in range(i+1, len(available)):
           j_idx = available[j]
           if masks_local[j_idx] & new_used:
               continue
           new_avail.append(j_idx)
   