    for i in range(4):
        u_entries = [(u(2,j), 1) for j in range(4)]
        v_entries = [(v(j,i), 1) for j in range(4)]
        w_entries = [(w(i,2), 1)]
        add_term(u_entries, v_entries, w_entries)
   