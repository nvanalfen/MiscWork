import numpy as np
import sys
from modular_alignments.vonmises_distribution import VonMisesHalf
from halotools.empirical_models.ia_models.ia_model_components import alignment_strength
from warnings import warn

class DimrothWatsonToVonMisesMapper:
    def __init__(self, primary_vm_params=None, secondary_vm_params=None, weight_params=None,
                 primary_vm_mapper=None, secondary_vm_mapper=None, weight_mapper=None):
        self.primary_vm_params = primary_vm_params
        self.secondary_vm_params = secondary_vm_params
        self.weight_params = weight_params
        # Set defaults from fitting if None are passed
        if primary_vm_params is None:
            self.primary_vm_params = np.array([0.83782722, 0.02517623, 0.09531156])
        if secondary_vm_params is None:
            self.secondary_vm_params = np.array([0.03301684, 0.06722184])
        if weight_params is None:
            self.weight_params = np.array([0.0408385, 0.05118375, 0.11909946])

        self.primary_vm_mapper = primary_vm_mapper
        self.secondary_vm_mapper = secondary_vm_mapper
        self.weight_mapper = weight_mapper
        # Set defaults functional forms if None are passed
        if primary_vm_mapper is None:
            self.primary_vm_mapper = lambda dw_mu, *params: np.polyval([*params, 0], dw_mu)
        if secondary_vm_mapper is None:
            self.secondary_vm_mapper = lambda dw_mu, *params: np.polyval([*params, 0], dw_mu)
        if weight_mapper is None:
            self.weight_mapper = lambda dw_mu, *params: np.polyval([*params], dw_mu)

        self.von_mises = VonMisesHalf()

    def _map_mu(self, dw_mu):
        primary_vm_mu = self.primary_vm_mapper(dw_mu, *self.primary_vm_params)
        secondary_vm_mu = self.secondary_vm_mapper(dw_mu, *self.secondary_vm_params)
        weight = self.weight_mapper(dw_mu, *self.weight_params)
        return primary_vm_mu, secondary_vm_mu, weight
    
    def pdf(self, x, dw_mu):
        primary_vm_mu, secondary_vm_mu, weight = self._map_mu(dw_mu)
        return weight*self.von_mises.pdf(x, alignment_strength(primary_vm_mu)) \
                + (1-weight)*self.von_mises.pdf(x, alignment_strength(secondary_vm_mu))
    
    def rvs(self, dw_mu, size=None, max_iter=100, random_state=None):
        dw_mu = np.atleast_1d(dw_mu)
        if size is None or size == ():
            size = len(dw_mu)
        if size != 1:
            # If size is an int, the first condition must be met, if size is a tuple, the second condition is the equivalent form
            if len(dw_mu) == size or dw_mu.shape == size:
                pass
            elif len(dw_mu) == 1:
                dw_mu = np.ones(size)*dw_mu
            else:
                msg = ('if `size` argument is given, len(dw_mu) must be 1 or equal to size.')
                raise ValueError(msg)
        else:
            size = len(dw_mu)
            
        # vector to store random variates
        result = np.zeros(size)

        # take care of dw_mu=0 case
        zero_mu = (dw_mu == 0)
        uran0 = np.random.rand(np.sum(zero_mu))*np.pi
        result[zero_mu] = uran0
        
        # take care of edge cases, i.e. mu is -1, 1
        result[dw_mu == 1.0] = np.random.choice([0.0,np.pi], size=np.sum(dw_mu == 1.0))
        result[dw_mu == -1.0] = np.pi/2

        # TODO: Work out the rejection sampling here

        # apply rejection sampling technique to sample from pdf
        edge_mask = ((dw_mu == 1.0) | (dw_mu == -1.0))
        n_sucess = np.sum(zero_mu) + np.sum(edge_mask)  # number of sucesessful draws from pdf
        n_iter = 0  # number of sample-reject iterations
        kk = dw_mu[(~zero_mu) & (~edge_mask)]  # store subset of k values that still need to be sampled
        mask = np.repeat(False,size)  # mask indicating which k values have a sucessful sample
        mask[zero_mu] = True
        mask[edge_mask] = True

        while (n_sucess < size) & (n_iter < max_iter):
            x_maxes = np.where( kk > 0, 0, np.pi/2 )                # x values for which the PDF will have its maximum value at a given mu
            y_maxes = self.pdf(x_maxes, kk)                         # maximum values to draw up to for each uniform pull
            x_draws = np.random.uniform(0, np.pi, size=len(kk))     # Random x values
            y_draws = np.random.uniform(0, y_maxes)                 # Randomly draw y values for each x, up to the maximum possible value for that PDF
            pdf = self.pdf(x_draws, kk)                             # Get the actual PDF for each drawn x using its respective mu

            keep = y_draws < pdf                                    # Keep it if the y_draw value is under the PDF curve at the given x_draw

            # count the number of succesful samples
            n_sucess += np.sum(keep)

            # store y values
            result[~mask] = x_draws                                 # Store all x values. Bad values will get overwritten in future steps

            # update mask indicating which values need to be redrawn
            mask[~mask] = keep

            # get subset of k values which need to be sampled.
            kk = kk[~keep]

            n_iter += 1

        if (n_iter == max_iter):
            msg = ('The maximum number of iterations reached, random variates may not be representative.')
            warn(msg)

        return result