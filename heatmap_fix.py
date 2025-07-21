#!/usr/bin/env python3
"""
Utility functions to fix heatmap display issues
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_wide_heatmap(data, x_labels=None, y_labels=None, title="Heatmap", 
                       figsize=(16, 8), annot=True, fmt='.2f', 
                       cmap='viridis', cbar=True, save_path=None):
    """
    Create a heatmap with extended horizontal size to show numerical values clearly
    
    Parameters:
    -----------
    data : 2D array
        Data to plot
    x_labels : list, optional
        Labels for x-axis
    y_labels : list, optional
        Labels for y-axis
    title : str
        Title of the plot
    figsize : tuple
        Figure size (width, height)
    annot : bool
        Whether to show numerical annotations
    fmt : str
        Format string for annotations (e.g., '.2f', '.1e')
    cmap : str
        Colormap name
    cbar : bool
        Whether to show colorbar
    save_path : str, optional
        Path to save the figure
    """
    
    # Create figure with extended width
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(data, 
                annot=annot, 
                fmt=fmt, 
                cmap=cmap, 
                cbar=cbar,
                xticklabels=x_labels,
                yticklabels=y_labels,
                ax=ax)
    
    # Set title
    ax.set_title(title, fontsize=14, pad=20)
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()
    return fig, ax

def create_compact_heatmap(data, x_labels=None, y_labels=None, title="Heatmap", 
                          figsize=(12, 8), annot=True, fmt='.1f', 
                          cmap='viridis', cbar=True, save_path=None):
    """
    Create a heatmap with compact text formatting to fit in smaller space
    
    Parameters:
    -----------
    data : 2D array
        Data to plot
    x_labels : list, optional
        Labels for x-axis
    y_labels : list, optional
        Labels for y-axis
    title : str
        Title of the plot
    figsize : tuple
        Figure size (width, height)
    annot : bool
        Whether to show numerical annotations
    fmt : str
        Format string for annotations (use compact format like '.1f')
    cmap : str
        Colormap name
    cbar : bool
        Whether to show colorbar
    save_path : str, optional
        Path to save the figure
    """
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with smaller text
    heatmap = sns.heatmap(data, 
                         annot=annot, 
                         fmt=fmt, 
                         cmap=cmap, 
                         cbar=cbar,
                         xticklabels=x_labels,
                         yticklabels=y_labels,
                         ax=ax,
                         annot_kws={'size': 8})  # Smaller annotation text
    
    # Set title
    ax.set_title(title, fontsize=12, pad=15)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()
    return fig, ax

def fix_existing_heatmap(fig, ax, fmt='.2f', annot_size=8):
    """
    Fix an existing heatmap by adjusting text formatting
    
    Parameters:
    -----------
    fig : matplotlib figure
        The figure containing the heatmap
    ax : matplotlib axis
        The axis containing the heatmap
    fmt : str
        Format string for annotations
    annot_size : int
        Font size for annotations
    """
    
    # Get the heatmap object
    heatmap = ax.collections[0]
    
    # Update annotation format and size
    if hasattr(heatmap, 'annotations'):
        for annotation in heatmap.annotations:
            # Update text format
            try:
                value = float(annotation.get_text())
                annotation.set_text(f'{value:{fmt}}')
            except ValueError:
                pass
            # Update font size
            annotation.set_fontsize(annot_size)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

def create_adaptive_heatmap(data, x_labels=None, y_labels=None, title="Heatmap", 
                           max_width=20, max_height=12, annot=True, 
                           cmap='viridis', cbar=True, save_path=None):
    """
    Create a heatmap with adaptive size based on data dimensions
    
    Parameters:
    -----------
    data : 2D array
        Data to plot
    x_labels : list, optional
        Labels for x-axis
    y_labels : list, optional
        Labels for y-axis
    title : str
        Title of the plot
    max_width : float
        Maximum figure width
    max_height : float
        Maximum figure height
    annot : bool
        Whether to show numerical annotations
    cmap : str
        Colormap name
    cbar : bool
        Whether to show colorbar
    save_path : str, optional
        Path to save the figure
    """
    
    # Calculate adaptive figure size based on data dimensions
    n_rows, n_cols = data.shape
    
    # Base size per cell
    cell_width = 0.8
    cell_height = 0.6
    
    # Calculate figure size
    fig_width = min(n_cols * cell_width + 2, max_width)  # +2 for margins
    fig_height = min(n_rows * cell_height + 2, max_height)
    
    # Choose format based on data range
    data_range = np.max(data) - np.min(data)
    if data_range > 1000:
        fmt = '.1e'  # Scientific notation for large numbers
    elif data_range > 10:
        fmt = '.1f'  # One decimal for medium numbers
    else:
        fmt = '.2f'  # Two decimals for small numbers
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    sns.heatmap(data, 
                annot=annot, 
                fmt=fmt, 
                cmap=cmap, 
                cbar=cbar,
                xticklabels=x_labels,
                yticklabels=y_labels,
                ax=ax,
                annot_kws={'size': max(6, min(12, 50 // max(n_rows, n_cols)))})
    
    # Set title
    ax.set_title(title, fontsize=12, pad=15)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()
    return fig, ax

# Example usage functions
def example_usage():
    """
    Example of how to use the heatmap functions
    """
    
    # Create sample data
    data = np.random.rand(5, 8) * 100
    
    print("=== Heatmap Display Solutions ===")
    print()
    print("1. Wide heatmap (extended horizontally):")
    print("   create_wide_heatmap(data, figsize=(16, 8))")
    print()
    print("2. Compact heatmap (smaller text):")
    print("   create_compact_heatmap(data, figsize=(12, 8), fmt='.1f')")
    print()
    print("3. Adaptive heatmap (auto-sized):")
    print("   create_adaptive_heatmap(data)")
    print()
    print("4. Fix existing heatmap:")
    print("   fix_existing_heatmap(fig, ax, fmt='.2f')")

if __name__ == "__main__":
    example_usage() 