from scipy.stats import beta, rv_continuous
import numpy as np

class MixedBetaDistribution(rv_continuous):
    """
    Custom distribution composed of the weighted average of two beta distribution.
    For speed, the percentile lookup function (_ppf) precomputes 10,000 percentile values
    and interpolates requested percentiles between these (Check performance note in _ppf).
    """
    
    def _pdf(self, x, w1, alpha1, beta1, alpha2, beta2):
        """PDF of the mixture"""
        w2 = 1 - w1
        return w1 * beta.pdf(x, alpha1, beta1) + w2 * beta.pdf(x, alpha2, beta2)
    
    def _cdf(self, x, w1, alpha1, beta1, alpha2, beta2):
        """CDF of the mixture"""
        w2 = 1 - w1
        return w1 * beta.cdf(x, alpha1, beta1) + w2 * beta.cdf(x, alpha2, beta2)
    
    def _ppf(self, q, w1, alpha1, beta1, alpha2, beta2):
        """
        Fast PPF using pre-computed lookup and interpolation
        
        Performance: This is able to compute values at 100k percentiles in 0.2 seconds as
        opposed to the built-in rv_continuous ppf which took roughly 210 seconds (3 minutes 30 seconds).
        The following metrics were obtained on a test of 100k draws on the range (0,1) using the values
        w1 = 0.283, alpha1 = 2.484, beta1 = 14.896, w2 = 0.717, alpha2 = 2.174, beta2 = 4.619 to
        fit the observed shape distribution in skysim5000.
        At worst, this lookup obtains a PDF value order 1e-6 away from the true value, with a mean
        difference of order 1e-8 and an MSE less than order 1e-13. The value determined to be that percentile
        is similarly close, with a worst case in testing differing by order 1e-7, with a mean difference
        of 1e-9 and an MSE of less than order 1e-7.
        """
        # Ensure parameters are scalars
        w1 = float(np.atleast_1d(w1).flat[0])
        alpha1 = float(np.atleast_1d(alpha1).flat[0])
        beta1 = float(np.atleast_1d(beta1).flat[0])
        alpha2 = float(np.atleast_1d(alpha2).flat[0])
        beta2 = float(np.atleast_1d(beta2).flat[0])
        
        # Build fine-grained lookup table with scalar parameters
        x_grid = np.linspace(0, 1, 10000)
        
        # Compute CDF on grid with scalar parameters
        w2 = 1 - w1
        beta1_dist = beta(alpha1, beta1)
        beta2_dist = beta(alpha2, beta2)
        cdf_grid = w1 * beta1_dist.cdf(x_grid) + w2 * beta2_dist.cdf(x_grid)
        
        # Use interpolation for inverse
        return np.interp(q, cdf_grid, x_grid)
    
    def _rvs(self, w1, alpha1, beta1, alpha2, beta2, size=None, random_state=None):
        """Generate random variates"""
        if size is None:
            size = 1
        
        # Use the provided random_state for reproducibility
        rng = self._random_state if random_state is None else random_state
        
        # Decide which component each sample comes from
        component = rng.uniform(size=size) < w1
        
        # Sample from each component
        samples = np.zeros(size)
        n_component1 = np.sum(component)
        n_component2 = size - n_component1
        
        if n_component1 > 0:
            samples[component] = beta.rvs(alpha1, beta1, size=n_component1, 
                                         random_state=rng)
        if n_component2 > 0:
            samples[~component] = beta.rvs(alpha2, beta2, size=n_component2, 
                                          random_state=rng)
        
        return samples

def get_empirical_percentiles(reference_data, query_values):
    """
    Find what percentile each query value would be in the reference distribution.
    
    Parameters:
    -----------
    reference_data : array
        The full distribution (e.g., all halo shapes)
    query_values : array
        Values to find percentiles for (e.g., galaxy-hosting halo shapes)
        
    Returns:
    --------
    percentiles : array
        Percentile (0-1) for each query value
    """
    # Sort the reference data
    sorted_ref = np.sort(reference_data)
    
    # Find where each query value would be inserted
    indices = np.searchsorted(sorted_ref, query_values, side='right')
    
    # Convert to percentiles (0 to 1)
    percentiles = indices / len(sorted_ref)
    
    return percentiles