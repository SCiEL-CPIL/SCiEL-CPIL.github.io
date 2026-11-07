<<<<<<< SEARCH
                c = np.clip(c, 1e-12, m)
                
                c_policy[t, z, i] = c
                
                # Compute value
                V_val = util(c)
                for zp in range(n_z):
                    prob = trans[z, zp]
                    if prob < 1e-10:
                        continue
                    m_next = R * (m - c) + income[zp]
                    m_next = np.clip(m_next, m_grid[0], m_grid[-1])
                    V_val += beta * prob * V_interp[zp](m_next)
                
                V_curr[z, i] = V_val
                Vm_curr[z, i] = util_prime(c)
        
        V_next = V_curr
        Vm_next = Vm_curr
        V_interp = build_pchip(V_next)
        Vm_interp = build_pchip(Vm_curr)
=======
                # This block is replaced by vectorized logic below
>>>>>>> REPLACE

Actually, it's cleaner to replace the entire `for z in range(n_z): for i...` block.

