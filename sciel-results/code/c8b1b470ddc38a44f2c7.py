                    for il in range(ni):
                        for jl in range(mj):
                            for kl in range(pk):
                                U_col = np.zeros(n*m)
                                V_col = np.zeros(m*p)
                                W_col = np.zeros(p*n)
                                
                                a_global = (n_starts[ib]+il)*5 + (m_starts[jb]+jl)
                                b_global = (m_starts[jb]+jl)*6 + (p_starts[kb]+kl)
                                c_global = (p_starts[kb]+kl)*4 + (n_starts[ib]+il)
                                
                                U_col[a_global] = 1
                                V_col[b_global] = 1
                                W_col[c_global] = 1
