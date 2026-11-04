                            orig_contrib = 0
                            for (i_prime, a_prime), u_val in u_nonzero.items():
                                if i_prime != i_merge:
                                    continue
                                for (a_prime2, k_prime), v_val in v_nonzero.items():
                                    if a_prime2 != a_prime:
                                        continue
                                    if k_prime != k_merge:
                                        continue
                                    if (k_prime, i_prime) in w_nonzero:
                                        orig_contrib += u_val * v_val * w_nonzero[(k_prime, i_prime)]
