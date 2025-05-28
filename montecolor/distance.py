"""NOTE: docstrings were written by Claude Sonnet 4. Cheers, Claude!"""

import numpy as np
import colorspacious as cs
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color


# The following addresses a DeprecationWarning when using `colormath` with
# recent versions of `numpy`. For details, see:
# https://github.com/gtaylor/python-colormath/issues/104
def patch_asscalar(a):
    return a.item()


setattr(np, "asscalar", patch_asscalar)


CVD_TYPES = (None, "deuteranomaly", "protanomaly", "tritanomaly")


def cvd_filter(color, cvd_type):
    """Apply color vision deficiency simulation to RGB colors.
    
    This function simulates how colors appear to individuals with different
    types of color vision deficiency (CVD) by applying colorspace
    transformations. It uses the colorspacious library to convert colors
    through a CVD simulation colorspace and back to standard RGB.
    
    Args:
        color (array_like): RGB color values to transform. Can be a single
            color or array of colors. Values should be in the range [0, 1]
            for sRGB1 colorspace.
        cvd_type (str or None): Type of color vision deficiency to simulate.
            Must be one of:
            - ``None``: No CVD simulation, returns original colors unchanged
            - ``"deuteranomaly"``: Reduced sensitivity to green light
                (most common)
            - ``"protanomaly"``: Reduced sensitivity to red light  
            - ``"tritanomaly"``: Reduced sensitivity to blue light (rarest)
    
    Returns:
        array_like: RGB color values as they would appear to someone with the
            specified color vision deficiency. Returns same shape as input.
            For ``cvd_type=None``, returns input unchanged.
    
    Note:
        - CVD simulation uses 100% severity for maximum effect
        - Input colors should be in sRGB1 format (range [0, 1])
        - Uses the colorspacious library's CVD simulation colorspace
        - Valid CVD types are defined in the module constant ``CVD_TYPES``
    
    Example:
        >>> import numpy as np
        >>> red = np.array([1.0, 0.0, 0.0])  # Pure red in sRGB1
        >>> deuteranomaly_red = cvd_filter(red, "deuteranomaly")
        >>> # Red as seen by someone with deuteranomaly (green deficiency)
        
        >>> colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> cvd_colors = cvd_filter(colors, "protanomaly")

    """
    if cvd_type is None:
        return color
    else:
        return cs.cspace_convert(
            color, dict(name="sRGB1+CVD", cvd_type=cvd_type, severity=100),
            "sRGB1"
        )


def distance_matrix(rgb_1, rgb_2):
    """Compute perceptual color distances across all CVD type combinations.
    
    This function calculates the CIE Delta E 2000 color difference between two
    RGB colors as perceived by individuals with different combinations of
    color vision deficiencies. It applies CVD filters in nested loops and
    computes perceptual distances in the Lab colorspace to assess color
    discriminability across the full spectrum of color vision types.
    
    Args:
        rgb_1 (array_like): First RGB color, shape ``(3,)`` with values in 
            range [0, 255].
        rgb_2 (array_like): Second RGB color, shape ``(3,)`` with values in 
            range [0, 255].
    
    Returns:
        numpy.ndarray: 3D array of Delta E distances with shape 
            ``(len(CVD_TYPES), len(CVD_TYPES), 1)``. Each element ``[i, j]``
            represents the perceptual distance between the two colors when
            ``rgb_1`` is viewed with ``CVD_TYPES[i]`` and ``rgb_2`` is viewed
            with ``CVD_TYPES[j]``.
    
    Note:
        - Colors are automatically normalized from [0, 255] to [0, 1] range
        - Uses CIE Delta E 2000 formula for perceptually uniform
            color differences
        - Converts colors through sRGB → Lab colorspace transformation
        - CVD filters are applied with 100% severity before
            distance calculation  
        - The nested CVD application allows comparison across different
            vision types
        - Higher Delta E values indicate greater perceptual color differences
    
    Example:
        >>> red = [255, 0, 0]
        >>> green = [0, 255, 0]
        >>> distances = distance_matrix(red, green)
        >>> print(distances.shape)
        (4, 4)
        >>> # distances[0, 0] = normal vision to normal vision
        >>> # distances[1, 2] = deuteranomaly to protanomaly comparison

    """
    rgb_1, rgb_2 = np.asarray(rgb_1) / 255., np.asarray(rgb_2) / 255.
    deltas = np.array([
        [
            delta_e_cie2000(
                convert_color(
                    sRGBColor(
                        *cvd_filter(
                            cvd_filter(rgb_1, cvd_a), cvd_b
                        ),
                        is_upscaled=False
                    ),
                    LabColor
                ),
                convert_color(
                    sRGBColor(
                        *cvd_filter(
                            cvd_filter(rgb_2, cvd_a), cvd_b
                        ),
                        is_upscaled=False
                    ),
                    LabColor
                ),
            ) for cvd_a in CVD_TYPES
        ] for cvd_b in CVD_TYPES
    ])
    return deltas


def cost_matrix(distance_matrix, index=-1):
    """Convert color distance matrix to cost matrix by a power law.
    
    This function transforms perceptual color distances into optimization
    costs by applying a power-law relationship. If ``index`` is negative,
    then smaller distances result in higher costs, encouraging the optimizer
    to avoid similar colors. Zero distances are replaced with a small epsilon
    value to prevent division by zero.

    Args:
        distance_matrix (array_like): Array of perceptual color distances, 
            typically Delta E values from ``distance_matrix()`` function.
        index (float, optional): Power-law exponent for the transformation.
            Defaults to ``-1``, which creates costs as ``cost = 1/distance``.
            More negative values create steeper penalties for small distances.
    
    Returns:
        numpy.ndarray: Cost matrix with same shape as input. Higher values
            indicate higher optimization costs (i.e., less desirable
            color combinations).
    
    Note:
        - Zero distances are replaced with ``1e-5`` to avoid division by zero
        - Default ``index=-1`` creates inverse relationship: smaller distances 
          yield higher costs
        - More negative indices (e.g., ``-2``) create stronger penalties for 
          close similarity, and weaker penalties for remote similarity
        - Positive indices would create direct relationships (not typical for 
          color optimization)

    Example:
        >>> distances = np.array([[0.0, 5.0], [10.0, 2.0]])
        >>> costs = cost_matrix(distances)

    """
    distance_matrix[distance_matrix == 0] = 1e-5
    return distance_matrix**index


def weighted_cost(rgb_1, rgb_2, weight_matrix=None, **kwargs):
    """Compute weighted average cost between two RGB colors across CVD types.
    
    This function calculates the optimization cost between two colors by
    computing perceptual distances across all color vision deficiency
    combinations, converting to costs, applying optional weighting, and
    returning the weighted average. This provides a single scalar cost value
    that accounts for color discrimination across different types of color
    vision.
    
    Args:
        rgb_1 (array_like): First RGB color, shape ``(3,)`` with values in 
            range [0, 255].
        rgb_2 (array_like): Second RGB color, shape ``(3,)`` with values in 
            range [0, 255].
        weight_matrix (array_like, optional): Weight matrix for different CVD
            type combinations, shape ``(len(CVD_TYPES), len(CVD_TYPES))``.
            If None, defaults to identity matrix (equal weighting for all 
            combinations). Higher weights emphasize specific CVD comparisons.
        **kwargs: Additional keyword arguments passed to ``cost_matrix()``,
            such as ``index`` for power-law exponent.
    
    Returns:
        float: Weighted average cost between the two colors. Higher values
            indicate the colors are more difficult to distinguish across
            the specified vision types, making them less desirable for
            palette optimization.
    
    Note:
        - Default identity weight matrix gives equal importance to all CVD 
          combinations
        - Custom weight matrices can emphasize specific vision types or 
          cross-comparisons
        - The cost aggregation provides a single objective value
          for optimization
        - Lower costs indicate better color discrimination across vision types
    
    Example:
        >>> red = [255, 0, 0]
        >>> green = [0, 255, 0] 
        >>> cost = weighted_cost(red, green)  # Equal weighting
        >>> print(f"Average cost: {cost:.3f}")
        
        >>> # Emphasize deuteranomaly comparisons
        >>> weights = np.identity(4)
        >>> weights[1, 1] = 2.0  # 2x weight for deuteranomaly-deuteranomaly
        >>> emphasized_cost = weighted_cost(red, green, weight_matrix=weights)

    """
    if weight_matrix is None:
        weight_matrix = np.identity(len(CVD_TYPES))
    cost = cost_matrix(distance_matrix(rgb_1, rgb_2), **kwargs)
    return np.mean(weight_matrix * cost)
