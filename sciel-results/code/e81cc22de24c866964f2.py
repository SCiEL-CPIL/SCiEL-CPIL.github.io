            for j in range(idx + 1, n_mods):
                m_j = moduli[j]
                if new_masks[j] != (1 << m_j) - 1:
                    new_masks[j] |= get_mask(m, r, m_j)
