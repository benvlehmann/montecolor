**Note:** This readme file was largely written by Claude Sonnet 4. Cheers, Claude!

# MonteColor

A Python package for generating colorblind-safe color palettes using Monte Carlo optimization and perceptual color science.

## Overview

MonteColor creates optimized color palettes that maintain excellent color discrimination for people with various types of color vision deficiency (CVD). The package uses advanced perceptual color models and optimization techniques to ensure your visualizations are accessible to the widest possible audience.

### Key Features

- **Colorblind-Safe Optimization**: Generates palettes optimized for deuteranomaly, protanomaly, and tritanomaly
- **Perceptual Color Science**: Uses CIE Delta E 2000 for perceptually uniform color differences  
- **Flexible Optimization**: Supports both MCMC sampling (via emcee) and direct optimization
- **Smart Color Sorting**: Arranges colors to maximize separation when using partial palettes
- **Customizable Constraints**: Support for fixed colors (backgrounds) and weighted optimization

## Installation

Clone the repository and install using `pip`:
```bash
git clone https://github.com/benvlehmann/montecolor.git
cd montecolor
pip install .
```

### Dependencies

- numpy
- scipy
- emcee
- colorspacious
- colormath
- seaborn

## Quick Start

```python
import numpy as np
from montecolor import Palette

# Create a 6-color palette including a white background as a fixed color
palette = Palette(6)

# Generate optimized colors using MCMC
optimized_palette = palette.generate(
    num_steps=200,
    mcmc=True,
    randomize_start=True
)

# Access the optimized colors
print("Fixed colors:", palette.fixed_colors)
print("Variable colors:", palette.variable_colors)

# Sort colors for partial palette usage
all_colors = np.vstack([palette.fixed_colors, palette.variable_colors])
sorted_colors = palette.sort(all_colors[1:])  # Exclude background
```

## Core Concepts

### Color Vision Deficiency Simulation

MonteColor simulates four vision types:
- **Normal vision** (trichromatic)
- **Deuteranomaly** (reduced green sensitivity, ~5% of males)
- **Protanomaly** (reduced red sensitivity, ~1% of males)  
- **Tritanomaly** (reduced blue sensitivity, <1% of population)

### Optimization Process

1. **Distance Calculation**: Computes perceptual distances using CIE Delta E 2000
2. **CVD Filtering**: Simulates how colors appear across different vision types
3. **Cost Computation**: Converts distances to optimization costs (typically cost = 1/distance)
4. **Weighted Aggregation**: Combines costs across vision types using configurable weights
5. **Optimization**: Uses MCMC or direct optimization to maximize color discrimination


## Advanced Usage

### Custom Optimization Parameters

```python
# Use direct optimization instead of MCMC
palette.generate(
    mcmc=False,
    minimize_kwargs={'method': 'L-BFGS-B'},
    randomize_start=True
)

# Fine-tune MCMC sampling
palette.generate(
    num_walkers=50,
    num_steps=500, 
    mcmc=True,
    sampler_kwargs=dict(
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    )
)
```

### Custom CVD Weighting

```python
# Emphasize deuteranomaly (most common CVD)
weight_matrix = np.identity(4)
weight_matrix[1, 1] = 2.0  # Double weight for deuteranomaly comparisons

palette = Palette(size=5, pair_matrix=weight_matrix)
```

### Color Distance Analysis

```python
from montecolor.distance import distance_matrix, cost_matrix, weighted_cost

# Analyze specific color pairs
red = [255, 0, 0]
green = [0, 255, 0]

# Get distances across all CVD combinations
distances = distance_matrix(red, green)
print(f"Distance matrix shape: {distances.shape}")  # (4, 4, 1)

# Convert to optimization costs
costs = cost_matrix(distances)

# Get weighted average cost
avg_cost = weighted_cost(red, green)
print(f"Average discrimination cost: {avg_cost:.3f}")
```

## Best Practices

### Choosing Palette Size
- **Small palettes (3-5 colors)**: Easier to optimize, better discrimination
- **Large palettes (8+ colors)**: May require more optimization steps or multiple runs

### Optimization Strategy
- **MCMC (default)**: Better global optimization, provides uncertainty estimates
- **Direct optimization**: Faster for simple cases, good for real-time applications

### Color Sorting
Always sort your final palette if users might use partial color sets:

```python
sorted_variable_colors = palette.sort(palette.variable_colors)
```

### Background Colors
Include background colors as fixed colors to ensure good contrast:

```python
# For dark backgrounds
palette = Palette(size=6, fixed_colors=[[0, 0, 0]])

# For multiple backgrounds  
palette = Palette(size=8, fixed_colors=[[255, 255, 255], [240, 240, 240]])
```

## Color Science Background

MonteColor is built on established color science principles:

- **sRGB Color Space**: Standard RGB with gamma correction
- **CIE Lab Color Space**: Perceptually uniform color representation
- **Delta E 2000**: State-of-the-art perceptual color difference formula
- **CVD Simulation**: Based on colorspacious library's physiologically-accurate models

The optimization maximizes the minimum pairwise color difference across all vision types, ensuring no two colors become indistinguishable for any form of color vision.

## License

MIT License - see LICENSE file for details.

## Citation

If you use MonteColor in academic work, please cite:

```bibtex
@software{montecolor,
  title={MonteColor},
  author={Benjamin V. Lehmann},
  year={2025},
  url={https://github.com/benvlehmann/montecolor}
}
```

## References

- Fairchild, M. D. (2013). *Color Appearance Models*. Wiley.
- Machado, G. M., Oliveira, M. M., & Fernandes, L. A. (2009). A physiologically-based model for simulation of color vision deficiency. *IEEE Transactions on Visualization and Computer Graphics*, 15(6), 1291-1298.
- Luo, M. R., Cui, G., & Rigg, B. (2001). The development of the CIE 2000 colour‐difference formula: CIEDE2000. *Color Research & Application*, 26(5), 340-350.
