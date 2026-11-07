            # Update pc (path for covariance)
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * ((m - m_old) / sigma)
            
            # hpc condition
            norm_pc = np.linalg.norm(pc)
            threshold_pc = (1 + 2.0 / D) * np.sqrt(1.0 - (1.0 - cc)**2) * np.sqrt(D)
            hpc = 1 if norm_pc / np.sqrt(1.0 - (1.0 - cc)**2) < threshold_pc else 0
            
            # Covariance update
            C = (1 - mueff * (c1 + cmu)) * C + c1 * (np.outer(pc, pc) + (1 - hpc) * cc * (2 - cc) * np.outer(pc, pc)) + cmu * sum_w
