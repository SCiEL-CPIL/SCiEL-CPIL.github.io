<<<<<<< SEARCH
        context = np.asarray(context)
        
        # 1. Update Global Model
        # A_global_new = gamma_global * A_global_old + x x^T
        # We update the inverse directly using SM formula
        self.inv_A_global = self._sm_update_inverse(self.inv_A_global, context, self.gamma_global)
        self.b_global = self.gamma_global * self.b_global + reward * context
=======
        context = np.asarray(context)
        
        # 1. Update Global Model with adaptive gamma
        # A_global_new = gamma_global * A_global_old + x x^T
        # We update the inverse directly using SM formula
        gamma_g = self._get_adaptive_gamma_global()
        self.inv_A_global = self._sm_update_inverse(self.inv_A_global, context, gamma_g)
        self.b_global = gamma_g * self.b_global + reward * context
>>>>>>> REPLACE

