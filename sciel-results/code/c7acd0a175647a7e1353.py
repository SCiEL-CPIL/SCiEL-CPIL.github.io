<<<<<<< SEARCH
        for i in range(len(S0)):
            for j in range(i+1, len(S0)):
            diff_mask0 |= (1 << abs(S0[j] - S0[i]))
=======
        for i in range(len(S0)):
            for j in range(i+1, len(S0)):
                diff_mask0 |= (1 << abs(S0[j] - S0[i]))
>>>>>>> REPLACE
