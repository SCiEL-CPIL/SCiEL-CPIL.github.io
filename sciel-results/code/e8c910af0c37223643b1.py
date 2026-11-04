Ur = Ua.reshape(n, m, r)  # (n, m, r)
Vr = Va.reshape(m, p, r)  # (m, p, r)  
Wr = Wa.reshape(p, n, r)  # (p, n, r)
T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
