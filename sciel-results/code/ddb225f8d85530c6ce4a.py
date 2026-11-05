<<<<<<< SEARCH
    def objective(c):
        max_err = 0.0
        for Om0, w0, wa in test_models:
            key = (Om0, w0, wa)
            f_refs, D_refs = refs[key]
            for i, a in enumerate(a_values):
                g = gamma_from_coeffs(c, a, Om0, w0, wa)
                if g < 0.1 or g > 1.2:
                    return 1e6
                
                f_app = f_approx(c, a, Om0, w0, wa)
                f_exact = f_refs[i]
                if abs(f_exact) > 1e-10:
                    err_f = abs(f_app - f_exact) / abs(f_exact)
                else:
                    err_f = 0.0
                
                D_app = D_approx(c, a, Om0, w0, wa)
                D_exact = D_refs[i]
                if abs(D_exact) > 1e-10:
                    err_D = abs(D_app - D_exact) / abs(D_exact)
                else:
                    err_D = 0.0
                
                max_err = max(max_err, err_f, err_D)
        
        return 100.0 * max_err
=======
    def objective(c):
        max_err = 0.0
        penalty = 0.0
        for Om0, w0, wa in test_models:
            key = (Om0, w0, wa)
            f_refs, D_refs = refs[key]
            for i, a in enumerate(a_values):
                g = gamma_from_coeffs(c, a, Om0, w0, wa)
                if g < 0.1 or g > 1.2:
                    # Heavy penalty for constraint violation
                    penalty += 1e6 * max(0.1 - g, g - 1.2)
                    continue # Skip error calculation for invalid points to save time? No, we need to know if it's valid.
                    # Actually, if penalty is huge, we return it. But let's accumulate.
                
                f_app = f_approx(c, a, Om0, w0, wa)
                f_exact = f_refs[i]
                if abs(f_exact) > 1e-10:
                    err_f = abs(f_app - f_exact) / abs(f_exact)
                else:
                    err_f = 0.0
                
                D_app = D_approx(c, a, Om0, w0, wa)
                D_exact = D_refs[i]
                if abs(D_exact) > 1e-10:
                    err_D = abs(D_app - D_exact) / abs(D_exact)
                else:
                    err_D = 0.0
                
                max_err = max(max_err, err_f, err_D)
        
        return 100.0 * max_err + penalty
>>>>>>> REPLACE

