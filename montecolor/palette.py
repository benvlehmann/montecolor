"""NOTE: docstrings were written by Claude Sonnet 4. Cheers, Claude!"""

import numpy as np
from scipy.optimize import minimize
import emcee
from seaborn.palettes import _ColorPalette

from .distance import weighted_cost


WHITE = np.array([255, 255, 255])


class Palette(object):
    """A color palette optimizer for generating colorblind-safe color schemes.
    
    This class represents a color palette that can be optimized to ensure good
    color discrimination for people with various types of color vision
    deficiency(CVD). The palette consists of both fixed colors (that remain
    constant during optimization) and variable colors (that are adjusted to
    maximize distinguishability).
    
    Args:
        size (int): Total number of colors in the palette, including both fixed
            and variable colors.
    
    Keyword Args:
        fixed_colors (list, optional): List of RGB color values that remain
            constant during optimization. Defaults to ``[WHITE]`` which
            typically represents the background color.
        cvd_matrix (array_like, optional): Transformation matrix for simulating 
            color vision deficiency types. Used to evaluate how the palette
            appears to people with different forms of colorblindness.
        index (float, optional): Power-law exponent for computing color
            discrimination costs as a function of perceptual distance. Default
            is ``-1``, corresponding to ``cost = 1/distance``.
        pair_matrix (array_like, optional): Weight matrix specifying the
            relative importance of discrimination between different color
            pairs. Shape should be ``(n_variable, n_fixed)``. Defaults to
            uniform weights.
    
    Attributes:
        size (int): Total number of colors in the palette.
        fixed_colors (numpy.ndarray): Array of fixed RGB colors,
            shape ``(n_fixed, 3)``.
        n_fixed (int): Number of fixed colors.
        n_variable (int): Number of variable colors to be optimized.
        n_dim (int): Total degrees of freedom (``3 * n_variable`` for
            RGB components).
        variable_colors (numpy.ndarray): Array of variable RGB colors,
            shape ``(n_variable, 3)``.
        cvd_matrix (array_like): Color vision deficiency transformation matrix.
        index (float): Power-law index for distance-based cost computation.
        pair_matrix (numpy.ndarray): Weights for color pair
            discrimination importance.
    
    Example:
        >>> palette = Palette(5, fixed_colors=[[1, 1, 1]], index=-2)
        >>> print(palette.n_variable)
        4
    """
    def __init__(self, size, **kwargs):
        # Set fixed colors that will not be modified during optimization
        # The background color should be included here (default white)
        self.fixed_colors = kwargs.get('fixed_colors', [WHITE])

        # Matrix of weights for color vision deficiency types
        self.cvd_matrix = kwargs.get('cvd_matrix')

        # Power-law index for computing costs as a function of color distance.
        # Default -1 corresponds to cost = 1/distance.
        self.index = kwargs.get('index', -1)

        # Total number of colors
        self.size = size

        self.fixed_colors = np.array(self.fixed_colors).reshape((-1, 3))
        self.n_fixed = len(self.fixed_colors)
        self.n_variable = self.size - self.n_fixed

        # Total number of degrees of freedom: 3 for each variable color
        self.n_dim = self.n_variable * 3

        self.variable_colors = np.zeros(self.n_dim).reshape((-1, 3))

        # Weights for different pairs (defaults to uniform)
        self.pair_matrix = kwargs.get(
            'pair_matrix', np.ones((self.n_variable, self.n_fixed)))

    @property
    def colors(self):
        return np.vstack((self.fixed_colors, self.variable_colors))

    def to_seaborn(self):
        return _ColorPalette(self.colors/255)

    def pairwise_cost(self, colors, variable_colors):
        matrix = np.zeros((len(variable_colors), len(colors)))
        for i, c1 in enumerate(variable_colors):
            for j, c2 in enumerate(colors):
                if i + len(colors) - len(variable_colors) != j:
                    matrix[i, j] = \
                        weighted_cost(c1, c2,
                                      weight_matrix=self.cvd_matrix,
                                      index=self.index)
        return matrix

    def cost(self, colors, variable_colors, pairs=True):
        cost_matrix = self.pairwise_cost(colors, variable_colors)
        if pairs:
            return np.sum(self.pair_matrix.T @ cost_matrix)
        else:
            return np.sum(cost_matrix)

    def set_variable_colors(self, variable_colors):
        self.variable_colors = variable_colors
        return self.cost(self.colors, self.variable_colors)

    def generate(
        self, num_walkers=None, num_steps=100, randomize_start=True,
        minimize_kwargs={}, mcmc=True, **sampler_kwargs
    ):
        """Generate optimized variable colors for the palette using MCMC
        or direct optimization.
    
        This method finds the optimal variable colors by maximizing color
        discrimination while considering color vision deficiency constraints.
        It supports two optimization approaches: Markov Chain Monte Carlo
        (MCMC) sampling using emcee, or direct optimization using
        scipy.optimize.minimize. MCMC is usually preferable when more than
        just a couple of colors are being varied.

        The method does not return the colors directly, but rather sets the
        ``variable_colors`` attribute and returns the associated cost.
        
        Keyword Args:
            num_walkers (int, optional): Number of MCMC walkers to use.
                If None, defaults to ``2 * n_dim`` where ``n_dim`` is the
                number of degrees of freedom.
            num_steps (int, optional): Number of MCMC steps to run.
                Defaults to 100.
            randomize_start (bool, optional): Whether to randomize initial
                positions. If True, starts from random positions within RGB
                bounds [0, 255]. If False, starts from current value of
                ``variable_colors``. You can use this to fine-tune an existing
                color palette. Defaults to True.
            minimize_kwargs (dict, optional): Additional keyword arguments
                passed to ``scipy.optimize.minimize`` when ``mcmc=False``.
                Defaults to empty dict.
            mcmc (bool, optional): Whether to use MCMC sampling (True) or
                direct optimization (False). Defaults to True.
            **sampler_kwargs: Additional keyword arguments passed to 
                ``emcee.EnsembleSampler`` when ``mcmc=True``.
        
        Returns:
            float: cost of the generated color palette.
        
        Note:
            - When using MCMC, the method creates an ``emcee.EnsembleSampler``
              stored in ``self.sampler`` for access to the full chain of
              MCMC samples.
            - When using direct optimization with ``mcmc=False``, multiple
              optimization runs are performed if ``randomize_start=True``, and
              the best result is selected.
            - All color values are constrained to the range [0, 255] using
              modular arithmetic.
            - The optimization maximizes the negative cost function to find
              colors with maximum discrimination.
        
        Example:
            >>> palette = Palette(5)
            >>> cost = palette.generate()

        """
        bounds = np.array([(0, 255)] * self.n_dim)

        def _log_prob(x):
            variable_colors = np.mod(x.reshape((-1, 3)), 255)
            colors = np.vstack((self.fixed_colors, variable_colors))
            return -self.cost(colors, variable_colors)

        if num_walkers is None:
            num_walkers = 2*self.n_dim

        if randomize_start:
            initial_positions = np.random.rand(num_walkers, self.n_dim)
            initial_positions = bounds[:, 0] + \
                initial_positions * (bounds[:, 1] - bounds[:, 0])
        else:
            initial_positions = self.variable_colors.reshape(-1)
        
        if mcmc:
            # Use MCMC for optimization
            self.sampler = emcee.EnsembleSampler(
                num_walkers, self.n_dim, _log_prob, **sampler_kwargs
            )
            self.sampler.run_mcmc(initial_positions, num_steps, progress=True)
            
            # Extract the samples and find the minimum value
            samples = self.sampler.get_chain(flat=True)
            best_index = np.argmax(self.sampler.get_log_prob(flat=True))
            best_params = samples[best_index]
        else:
            # use scipy.optimize.minimize
            guesses = initial_positions
            if not randomize_start:
                guesses = [guesses]
            results = [minimize(_log_prob, guess, **minimize_kwargs)
                       for guess in guesses]
            best_params = results[
                np.argmin([r.fun for r in results])
            ].x

        best_params = np.mod(best_params, 255)
        return self.set_variable_colors(best_params.reshape((-1, 3)))

    def sort(self, cs):
        """Sort colors to maximize separation when using partial palettes.
    
        This method reorders colors using a recursive algorithm that places
        the most similar colors at opposite ends of the sequence. This
        ensures that if only a subset of the palette is used (e.g., first N
        colors), very similar colors are less likely to be selected together,
        maintaining good color discrimination.
        
        The sorting algorithm works as follows:
        1. Compute pairwise distances between all variable colors
        2. Identify the pair with the highest cost (most similar)
        3. Place these two colors at opposite ends of the sequence
        4. Recursively sort the remaining n-2 colors for the middle section
        5. Determine optimal orientation by minimizing connection costs
           between outer colors and the sorted middle section
        
        Args:
            cs (array_like): Array of colors to sort, shape ``(n_colors, 3)``
                where each row represents an RGB color.
        
        Returns:
            list: Sorted list of colors arranged to maximize separation when
                using partial palettes. Colors are returned in the same format
                as input.
        
        Note:
            - For palettes with 2 or fewer colors, returns the input unchanged
            - This is particularly useful when users may only use the first
              few colors from a larger palette
        
        Example:
            >>> palette = Palette(5)
            >>> colors = [[255, 0, 0], [250, 0, 0], [0, 255, 0], [0, 0, 255]]
            >>> sorted_colors = palette.sort(colors)

        """
        if len(cs) <= 2:
            return cs
        else:
            cost_matrix = self.pairwise_cost(cs, cs)
            i1, i2 = np.array(np.where(cost_matrix == cost_matrix.max())).T[0]
            c1, c2 = cs[i1], cs[i2]
            middle = [c for i, c in enumerate(cs) if i not in (i1, i2)]
            middle = self.sort(middle)
            # Figure out which way to attach the outer two
            cost_1 = self.cost([c1], [middle[0]], pairs=None) + \
                self.cost([c2], [middle[-1]], pairs=None)
            cost_2 = self.cost([c2], [middle[0]], pairs=None) + \
                self.cost([c1], [middle[-1]], pairs=None)
            if cost_1 < cost_2:
                return [c1] + middle + [c2]
            else:
                return [c2] + middle + [c1]
