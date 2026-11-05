<<<<<<< SEARCH
# Factor 0: M1
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    # U: A00, A11
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    ...
=======
# Factor 0: M1
u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
# U: A00, A11
u_vec[(off_a+0)*5 + (off_b+0)] = 1
u_vec[(off_a+1)*5 + (off_b+1)] = 1
...
>>>>>>> REPLACE
