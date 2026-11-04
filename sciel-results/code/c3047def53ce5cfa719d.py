       for i_start, i_size in i_groups:
           for j_start, j_size in j_groups:
               for k_start, k_size in k_groups:
                   if i_size == 2 and j_size == 2 and k_size == 2:
                       strassen_2x2x2(...)
                   else:
                       schoolbook(...)
       