import numpy as np
from scipy.special import comb
from scipy.stats import invgamma, multivariate_normal


class BayesianLinearRegression:
    """
    Bayesian weighted linear regression with:
        y ~ N(X Phi, sigma^2 W^{-1})
        Phi | sigma^2 ~ N(0, sigma^2 I)   
    """

    def __init__(self):
        self.Phi_hat = None # posterior mean of Phi (intercept + coefs)
        self.V_Phi = None # posterior covariance factor
        self.s2 = None  # residual variance
        self.N = None   # number of samples
        self.D = None  # number of coefficients (d + 1)

    def fit(self, X, y, weights):
        """
        Fit Bayesian linear regression.

        PARAMETERS (follow phase 2 definitions)
        X: array, shape (N, D)
        y: array, shape (N,)
            model outputs under perturbations.
        weights: Locality weights (with SHAP kernel)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(weights, dtype=float)

        N, D = X.shape
        self.N = N
        self.D = D

        # Diagonal weight matrix W
        W = np.diag(w)

        # X^T W X
        XTWX = X.T @ W @ X

        self.V_Phi = np.linalg.inv(XTWX + np.eye(D))
        #self.V_Phi = np.linalg.inv(XTWX)

        # Posterior mean
        self.Phi_hat = self.V_Phi @ X.T @ W @ y

        # Residuals
        r = y - X @ self.Phi_hat

        # Empirical residual variance s^2 (matches Eq 7)
        self.s2 = (r.T @ W @ r + self.Phi_hat.T @ self.Phi_hat) / N

    def draw_posterior_samples(self, num_samples):
        """
        Draw samples from the posterior distributions of Phi and sigma^2.
        These are defined in paper

        Returns
        Phi_samples: Samples of regression coefficients (intercept + d feature coefs).
        sigma2_samples: Samples of noise variance.
        """

        # Posterior of sigma^2 under uninformative prior:
        # sigma^2 | data ~ InvGamma(alpha = N/2, beta = N s^2 / 2)
        alpha = self.N / 2.0
        beta = (self.N * self.s2) / 2.0
        sigma2_samples = invgamma.rvs(alpha, scale=beta, size=num_samples)

        # For each sigma^2, sample Phi ~ N(Phi_hat, sigma^2 V_Phi)
        Phi_samples = np.zeros((num_samples, self.D))
        for i in range(num_samples):
            cov = self.V_Phi * sigma2_samples[i]
            Phi_samples[i] = multivariate_normal.rvs(
                mean=self.Phi_hat, cov=cov
            )

        return Phi_samples, sigma2_samples


class BayesSHAPTabular:
    """
    BayesSHAP for tabular classification with SHAP kernel

    classifier with predict_proba(X) -> (n_samples, n_classes).
    """

    def __init__(self, model, background, n_perturb=2048,
                 n_posterior=1000, cred_level=0.95, random_state=None):
        """
        PARAMS
        model : Classifier with predict_proba method (NOT the function)
        background : array, shape (n_bg, d)
            For comparison sake, input should be exact same as KernelSHAP
        n_perturb : number of perturbations N
        n_posterior : Number of posterior samples B.
        cred_level : Credible level (ex: 0.95 for 95% credible intervals)
        random_state 
        """
        self.model = model
        self.background = np.asarray(background, dtype=float)
        self.n_perturb = n_perturb
        self.n_posterior = n_posterior
        self.cred_level = cred_level
        self.rng = np.random.default_rng(random_state)

    # 
    # SHAP kernel and coalition sampling 

    @staticmethod
    def shap_kernel(z):
        """
        Shapley locality weight for a single perturbation vec z in {0,1}^d.
        pi(z) = (d-1) / [|z| (d - |z|) * C(d, |z|)]  for 1 <= |z| <= d-1
        """
        z = np.asarray(z, dtype=int)
        d = z.size
        m = int(z.sum())
        if m == 0 or m == d:
            # KernelSHAP uses infinite weight; we avoid those coalitions.
            return 0.0
        return (d - 1.0) / (m * (d - m) * comb(d, m))

    def _sample_coalitions(self, d, N):
        """
        Sample N coalitions z in {0,1}^d with 1 <= |z| <= d-1.
        Returns Z matrix of shape (N, d).
        """
        Z = np.zeros((N, d), dtype=int)
        for i in range(N):
            # choose coalition size uniformly from 1..d-1
            m = self.rng.integers(1, d)
            idx = self.rng.choice(d, size=m, replace=False)
            z = np.zeros(d, dtype=int)
            z[idx] = 1
            Z[i] = z
        return Z

    def _perturb(self, x, z):
        """
        Construct perturbed sample by combining x with background instance according to z in {0,1}^d.

        For feature j:
            if z_j = 1 -> use x_j
            if z_j = 0 -> use background_j
        """
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=int)
        d = x.size
        # sample a single background row
        bg = self.background[self.rng.integers(0, self.background.shape[0])]
        out = bg.copy()
        out[z == 1] = x[z == 1]
        return out

    # Main interface 

    def explain(self, x, label=1):
        """
        Compute BayesSHAP explanation for feature vec x and class index `label`.

        PARAMS
        x: array, shape (d,)
        label: int
            Class index for which to explain predict_proba(x)[label].

        Returns
        result: as a dict:
            -'phi_mean': shape (d,)
                Posterior mean of feature contributions.
            -'phi_samples': shape (B, d)
                Posterior samples of feature contributions.
            -'phi_ci_lower': shape (d,)
                Lower credible bound 
            -'phi_ci_upper': (d,)
                Upper credible bound
            -'baseline': float
                Intercept/baseline term (mean of posterior intercept).
        """
        x = np.asarray(x, dtype=float)
        d = x.size
        N = self.n_perturb
        B = self.n_posterior

        # Phase 1: Generate perturbations and weights 
        # Sample coalitions
        Z = self._sample_coalitions(d, N)  # shape (N, d)

        # perturbed inputs f(perturbed x)
        X_pert = np.zeros((N, d), dtype=float)
        y = np.zeros(N, dtype=float)
        weights = np.zeros(N, dtype=float)

        for i in range(N):
            z = Z[i]
            X_pert[i] = self._perturb(x, z)
            # predict probability for given label
            proba = self.model.predict_proba(X_pert[i].reshape(1, -1))[0]
            y[i] = proba[label]
            weights[i] = self.shap_kernel(z)

        # Remove any zero-weight rows (can happen if |z|=0 or d)
        mask = weights > 0
        Z = Z[mask]
        X_pert = X_pert[mask]
        y = y[mask]
        weights = weights[mask]
        N_eff = Z.shape[0]

        if N_eff == 0:
            raise RuntimeError("No valid coalitions after weighting; "
                               "check perturbation/coalition settings.")

        # Phase 2: Bayesian weighted linear regression 
        # Matrix with intercept: X = [1, Z]
        X_design = np.concatenate([np.ones((N_eff, 1)), Z], axis=1)

        blr = BayesianLinearRegression()
        blr.fit(X_design, y, weights)

        # Phase 3: Posterior sampling 
        Phi_samples, sigma2_samples = blr.draw_posterior_samples(B)
        # Discard intercept column for feature contributions
        phi_samples = Phi_samples[:, 1:]  # shape (B, d)
        baseline_samples = Phi_samples[:, 0]  # intercept samples

        # Phase 4: Summary statistics + credible intervals 
        phi_mean = phi_samples.mean(axis=0)
        baseline_mean = baseline_samples.mean()

        alpha = 1.0 - self.cred_level
        lower_q = 100 * (alpha / 2.0)
        upper_q = 100 * (1.0 - alpha / 2.0)

        phi_ci_lower = np.percentile(phi_samples, lower_q, axis=0)
        phi_ci_upper = np.percentile(phi_samples, upper_q, axis=0)

        return {
            "phi_mean": phi_mean,
            "phi_samples": phi_samples,
            "phi_ci_lower": phi_ci_lower,
            "phi_ci_upper": phi_ci_upper,
            "baseline": baseline_mean,
            "sigma2_samples": sigma2_samples,
        }
