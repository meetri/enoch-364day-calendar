"""
Publication Style Configuration for Multiple Journal Submissions

Supported Templates:
1. SAGE Journal (Journal for the History of Astronomy)
   - Single-column width: 8.5cm = 3.35 inches
   - Figures: fig0X_<name>_v2.pdf

2. AHES (Archive for History of Exact Sciences)
   - Full-width column: 6.5 inches
   - Figures: fig0X_<name>_ahes.pdf

3. arXiv Preprint
   - Full-width column: 6.5 inches (same as AHES)
   - Figures: Reuses AHES figures (fig0X_<name>_ahes.pdf)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import numpy as np

# SAGE Journal Specifications
COLUMN_WIDTH_INCHES = 3.35  # 8.5cm single column
DOUBLE_COLUMN_WIDTH_INCHES = 7.0  # 17.5cm double column
DPI = 300

# AHES (Archive for History of Exact Sciences) Specifications
AHES_FULL_WIDTH_INCHES = 6.5  # Full column width for AHES
AHES_DPI = 300

# arXiv Preprint Specifications (aliases to AHES - same dimensions)
ARXIV_FULL_WIDTH_INCHES = AHES_FULL_WIDTH_INCHES  # 6.5 inches (same as AHES)
ARXIV_DPI = AHES_DPI  # Same resolution

# Typography - SAGE
FONT_FAMILY = 'Times New Roman'
FONT_SIZE_LABELS = 9
FONT_SIZE_TICKS = 8
FONT_SIZE_LEGEND = 8
FONT_SIZE_ANNOTATIONS = 8

# Typography - AHES (scaled for larger figures)
AHES_FONT_SIZE_LABELS = 11
AHES_FONT_SIZE_TICKS = 10
AHES_FONT_SIZE_LEGEND = 10
AHES_FONT_SIZE_ANNOTATIONS = 9

# Colorblind-safe palette (Wong 2011)
COLORS = {
    'blue': '#0173B2',      # Primary line
    'orange': '#DE8F05',    # Secondary line
    'green': '#029E73',     # Tertiary line
    'purple': '#CC78BC',    # Accent
    'red': '#CA3542',       # Negative/fail
    'yellow': '#ECE133',    # Highlight
    'cyan': '#56B4E9',      # Alternative blue
    'black': '#000000',
    'gray': '#808080',
    'light_gray': '#E0E0E0'
}

# Accuracy band colors (green to yellow gradient)
ACCURACY_COLORS = {
    'excellent': '#00A087',   # Dark green for ±0.5°
    'good': '#3CB371',        # Medium green for ±1.0°
    'fair': '#90EE90',        # Light green for ±3.0°
    'poor': '#F9E79F'         # Yellow for outside bounds
}

# Line and marker specifications
LINE_WIDTH_PRIMARY = 1.5
LINE_WIDTH_SECONDARY = 1.0
LINE_WIDTH_GRID = 0.5
LINE_WIDTH_AXES = 0.75
MARKER_SIZE = 4

def set_publication_style():
    """
    Configure matplotlib with SAGE journal specifications.
    Call this at the beginning of each notebook.
    """
    # Set the style
    plt.style.use('seaborn-v0_8-paper')

    # Configure rcParams
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    rcParams['font.size'] = FONT_SIZE_TICKS
    rcParams['axes.labelsize'] = FONT_SIZE_LABELS
    rcParams['axes.titlesize'] = FONT_SIZE_LABELS
    rcParams['xtick.labelsize'] = FONT_SIZE_TICKS
    rcParams['ytick.labelsize'] = FONT_SIZE_TICKS
    rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
    rcParams['figure.dpi'] = DPI
    rcParams['savefig.dpi'] = DPI
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.05
    rcParams['savefig.transparent'] = False
    rcParams['savefig.format'] = 'pdf'

    # Axes styling
    rcParams['axes.linewidth'] = LINE_WIDTH_AXES
    rcParams['axes.edgecolor'] = COLORS['black']
    rcParams['axes.labelcolor'] = COLORS['black']
    rcParams['axes.grid'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False

    # Grid styling
    rcParams['grid.color'] = COLORS['light_gray']
    rcParams['grid.linewidth'] = LINE_WIDTH_GRID
    rcParams['grid.alpha'] = 0.5

    # Tick styling
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = LINE_WIDTH_AXES
    rcParams['ytick.major.width'] = LINE_WIDTH_AXES
    rcParams['xtick.color'] = COLORS['black']
    rcParams['ytick.color'] = COLORS['black']

    # Legend styling
    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 0.8
    rcParams['legend.edgecolor'] = COLORS['light_gray']
    rcParams['legend.fancybox'] = False

    # Line and marker defaults
    rcParams['lines.linewidth'] = LINE_WIDTH_PRIMARY
    rcParams['lines.markersize'] = MARKER_SIZE

    # PDF specific
    rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable)
    rcParams['ps.fonttype'] = 42

def create_figure(width=COLUMN_WIDTH_INCHES, height=None, aspect_ratio=1.2):
    """
    Create a figure with publication specifications.

    Parameters:
    -----------
    width : float
        Figure width in inches (default: single column = 3.35")
    height : float or None
        Figure height in inches. If None, calculated from aspect_ratio
    aspect_ratio : float
        Height/width ratio if height not specified

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    if height is None:
        height = width / aspect_ratio

    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax

def create_multipanel_figure(nrows=2, ncols=1, width=COLUMN_WIDTH_INCHES,
                              height=None, aspect_ratio=1.2, **kwargs):
    """
    Create a multi-panel figure with publication specifications.

    Parameters:
    -----------
    nrows, ncols : int
        Number of rows and columns
    width : float
        Total figure width in inches
    height : float or None
        Total figure height in inches
    aspect_ratio : float
        Overall height/width ratio if height not specified
    **kwargs : additional arguments passed to plt.subplots

    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    if height is None:
        height = width / aspect_ratio

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)
    return fig, axes

def save_publication_figure(fig, filename, format='pdf', dpi=DPI,
                            tight=True, transparent=False):
    """
    Save figure with publication specifications.

    Parameters:
    -----------
    fig : matplotlib figure
        The figure to save
    filename : str
        Output filename (with or without extension)
    format : str
        Output format ('pdf', 'png', 'eps')
    dpi : int
        Resolution for raster formats
    tight : bool
        Use tight bounding box
    transparent : bool
        Transparent background
    """
    # Ensure proper extension
    if not filename.endswith(f'.{format}'):
        filename = f'{filename}.{format}'

    bbox = 'tight' if tight else None

    fig.savefig(filename,
                format=format,
                dpi=dpi,
                bbox_inches=bbox,
                pad_inches=0.05,
                transparent=transparent,
                facecolor='white' if not transparent else 'none',
                edgecolor='none')

    print(f"Figure saved: {filename}")
    print(f"  Format: {format.upper()}, DPI: {dpi}")
    print(f"  Size: {fig.get_figwidth():.2f}\" × {fig.get_figheight():.2f}\"")

def format_year_axis(ax, start_year, end_year, use_bce_ce=True):
    """
    Format x-axis for year display with BCE/CE notation.

    Parameters:
    -----------
    ax : matplotlib axes
        The axes to format
    start_year : int
        Starting year (negative for BCE)
    end_year : int
        Ending year
    use_bce_ce : bool
        If True, use BCE/CE notation; if False, use plain numbers
    """
    if use_bce_ce:
        # Custom formatter for BCE/CE
        def year_formatter(x, pos):
            if x < 0:
                return f'{int(-x)} BCE'
            elif x > 0:
                return f'{int(x)} CE'
            else:
                return '0'

        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(year_formatter))

    ax.set_xlim(start_year, end_year)

def add_accuracy_bands(ax, y_limits=(-3, 5), alpha=0.2, add_labels=False):
    """
    Add colored bands showing accuracy tiers.

    Parameters:
    -----------
    ax : matplotlib axes
        The axes to add bands to
    y_limits : tuple
        (ymin, ymax) for band extent
    alpha : float
        Band transparency
    add_labels : bool
        If True, add labels to bands for legend (default: False)
    """
    # ±0.5° band (excellent)
    ax.axhspan(-0.5, 0.5, color=ACCURACY_COLORS['excellent'],
               alpha=alpha, zorder=0,
               label='±0.5°' if add_labels else '')

    # ±0.5° to ±1.0° band (good)
    ax.axhspan(-1.0, -0.5, color=ACCURACY_COLORS['good'],
               alpha=alpha, zorder=0)
    ax.axhspan(0.5, 1.0, color=ACCURACY_COLORS['good'],
               alpha=alpha, zorder=0,
               label='±1.0°' if add_labels else '')

    # ±1.0° to ±3.0° band (fair)
    ax.axhspan(-3.0, -1.0, color=ACCURACY_COLORS['fair'],
               alpha=alpha, zorder=0)
    ax.axhspan(1.0, 3.0, color=ACCURACY_COLORS['fair'],
               alpha=alpha, zorder=0,
               label='±3.0°' if add_labels else '')

    ax.set_ylim(y_limits)

def add_grid(ax, which='major', alpha=0.3):
    """
    Add subtle grid to axes.

    Parameters:
    -----------
    ax : matplotlib axes
        The axes to add grid to
    which : str
        'major', 'minor', or 'both'
    alpha : float
        Grid transparency
    """
    ax.grid(True, which=which, alpha=alpha,
            color=COLORS['light_gray'],
            linewidth=LINE_WIDTH_GRID)

def annotate_point(ax, x, y, text, xytext=(10, 10), **kwargs):
    """
    Add an annotation with consistent styling.

    Parameters:
    -----------
    ax : matplotlib axes
        The axes to annotate
    x, y : float
        Point to annotate
    text : str
        Annotation text
    xytext : tuple
        Offset in points
    **kwargs : additional arguments for ax.annotate
    """
    defaults = {
        'fontsize': FONT_SIZE_ANNOTATIONS,
        'bbox': dict(boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor=COLORS['light_gray'],
                    alpha=0.8),
        'arrowprops': dict(arrowstyle='->',
                          connectionstyle='arc3,rad=0.2',
                          color=COLORS['black'],
                          lw=LINE_WIDTH_SECONDARY)
    }
    defaults.update(kwargs)

    ax.annotate(text, xy=(x, y), xytext=xytext,
                textcoords='offset points', **defaults)

def set_ahes_style():
    """
    Configure matplotlib with AHES (Archive for History of Exact Sciences) specifications.
    Similar to SAGE style but with larger fonts for full-width figures.
    Call this at the beginning of AHES figure generation.
    """
    # Set the base style
    plt.style.use('seaborn-v0_8-paper')

    # Configure rcParams with AHES font sizes
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    rcParams['font.size'] = AHES_FONT_SIZE_TICKS
    rcParams['axes.labelsize'] = AHES_FONT_SIZE_LABELS
    rcParams['axes.titlesize'] = AHES_FONT_SIZE_LABELS
    rcParams['xtick.labelsize'] = AHES_FONT_SIZE_TICKS
    rcParams['ytick.labelsize'] = AHES_FONT_SIZE_TICKS
    rcParams['legend.fontsize'] = AHES_FONT_SIZE_LEGEND
    rcParams['figure.dpi'] = AHES_DPI
    rcParams['savefig.dpi'] = AHES_DPI
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.05
    rcParams['savefig.transparent'] = False
    rcParams['savefig.format'] = 'pdf'

    # Axes styling (same as SAGE)
    rcParams['axes.linewidth'] = LINE_WIDTH_AXES
    rcParams['axes.edgecolor'] = COLORS['black']
    rcParams['axes.labelcolor'] = COLORS['black']
    rcParams['axes.grid'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False

    # Grid styling (same as SAGE)
    rcParams['grid.color'] = COLORS['light_gray']
    rcParams['grid.linewidth'] = LINE_WIDTH_GRID
    rcParams['grid.alpha'] = 0.5

    # Tick styling (same as SAGE)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = LINE_WIDTH_AXES
    rcParams['ytick.major.width'] = LINE_WIDTH_AXES
    rcParams['xtick.color'] = COLORS['black']
    rcParams['ytick.color'] = COLORS['black']

    # Legend styling (same as SAGE)
    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 0.8
    rcParams['legend.edgecolor'] = COLORS['light_gray']
    rcParams['legend.fancybox'] = False

    # Line and marker defaults (same as SAGE)
    rcParams['lines.linewidth'] = LINE_WIDTH_PRIMARY
    rcParams['lines.markersize'] = MARKER_SIZE

    # PDF specific (same as SAGE)
    rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable)
    rcParams['ps.fonttype'] = 42

def create_ahes_figure(width=AHES_FULL_WIDTH_INCHES, height=None, aspect_ratio=1.2):
    """
    Create a figure with AHES specifications (full-width technical version).

    Parameters:
    -----------
    width : float
        Figure width in inches (default: AHES full width = 6.5")
    height : float or None
        Figure height in inches. If None, calculated from aspect_ratio
    aspect_ratio : float
        Height/width ratio if height not specified

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    if height is None:
        height = width / aspect_ratio

    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax

def create_ahes_multipanel_figure(nrows=2, ncols=1, width=AHES_FULL_WIDTH_INCHES,
                                   height=None, aspect_ratio=1.2, **kwargs):
    """
    Create a multi-panel figure with AHES specifications.

    Parameters:
    -----------
    nrows, ncols : int
        Number of rows and columns
    width : float
        Total figure width in inches (default: AHES full width)
    height : float or None
        Total figure height in inches
    aspect_ratio : float
        Overall height/width ratio if height not specified
    **kwargs : additional arguments passed to plt.subplots

    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    if height is None:
        height = width / aspect_ratio

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)
    return fig, axes

def save_ahes_figure(fig, filename, format='pdf', dpi=AHES_DPI,
                     tight=True, transparent=False):
    """
    Save figure with AHES specifications.
    Adds '_ahes' suffix before extension if not already present.

    Parameters:
    -----------
    fig : matplotlib figure
        The figure to save
    filename : str
        Output filename (with or without extension)
    format : str
        Output format ('pdf', 'png', 'eps')
    dpi : int
        Resolution for raster formats
    tight : bool
        Use tight bounding box
    transparent : bool
        Transparent background
    """
    # Add _ahes suffix if not present
    if '_ahes' not in filename:
        base = filename.replace(f'.{format}', '')
        filename = f'{base}_ahes'

    # Ensure proper extension
    if not filename.endswith(f'.{format}'):
        filename = f'{filename}.{format}'

    bbox = 'tight' if tight else None

    fig.savefig(filename,
                format=format,
                dpi=dpi,
                bbox_inches=bbox,
                pad_inches=0.05,
                transparent=transparent,
                facecolor='white' if not transparent else 'none',
                edgecolor='none')

    print(f"AHES Figure saved: {filename}")
    print(f"  Format: {format.upper()}, DPI: {dpi}")
    print(f"  Size: {fig.get_figwidth():.2f}\" × {fig.get_figheight():.2f}\"")

# Example usage template
if __name__ == '__main__':
    # Set the publication style
    set_publication_style()

    # Create a test figure
    fig, ax = create_figure()

    # Sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Plot with publication style
    ax.plot(x, y, color=COLORS['blue'], linewidth=LINE_WIDTH_PRIMARY,
            label='Sample data')

    # Add grid
    add_grid(ax)

    # Labels
    ax.set_xlabel('X axis label')
    ax.set_ylabel('Y axis label')
    ax.legend()

    # Save
    save_publication_figure(fig, 'test_figure')

    plt.show()
