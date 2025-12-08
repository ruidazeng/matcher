#!/usr/bin/env python3
"""
Generate visualization images for the README.
These diagrams reflect the actual code structure in psig_matcher/.

Run from the project root:
    python scripts/generate_readme_images.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Create images directory
IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# Colors
BLUE = '#2563EB'
PURPLE = '#7C3AED'
GREEN = '#059669'
ORANGE = '#D97706'
DARK = '#1F2937'
LIGHT = '#F3F4F6'


def generate_concept_diagram():
    """
    Shows the actual workflow from utils.py:
    1. Load Part data
    2. Get PartInstance  
    3. Get PiezoelectricSignature
    4. Compare with Comparator (MSE, RMSE, L1)
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    ax.text(6, 4.5, 'Piezoelectric Signature Matching Workflow', 
            fontsize=14, fontweight='bold', ha='center', color=DARK)
    
    # Step 1: Part
    ax.add_patch(FancyBboxPatch((0.5, 1.5), 2.2, 2, boxstyle="round,pad=0.1",
                                 facecolor=BLUE, edgecolor=DARK, linewidth=2))
    ax.text(1.6, 3.1, 'Part', fontsize=11, fontweight='bold', ha='center', color='white')
    ax.text(1.6, 2.5, 'e.g. "SEN"', fontsize=9, ha='center', color='white')
    ax.text(1.6, 2.0, 'Part(part_type)', fontsize=8, ha='center', color='white', family='monospace')
    
    # Step 2: PartInstance
    ax.add_patch(FancyBboxPatch((3.4, 1.5), 2.2, 2, boxstyle="round,pad=0.1",
                                 facecolor=PURPLE, edgecolor=DARK, linewidth=2))
    ax.text(4.5, 3.1, 'PartInstance', fontsize=11, fontweight='bold', ha='center', color='white')
    ax.text(4.5, 2.5, 'e.g. "x1"', fontsize=9, ha='center', color='white')
    ax.text(4.5, 2.0, 'get_instance(id)', fontsize=8, ha='center', color='white', family='monospace')
    
    # Step 3: PiezoelectricSignature
    ax.add_patch(FancyBboxPatch((6.3, 1.5), 2.6, 2, boxstyle="round,pad=0.1",
                                 facecolor=GREEN, edgecolor=DARK, linewidth=2))
    ax.text(7.6, 3.1, 'Signature', fontsize=11, fontweight='bold', ha='center', color='white')
    ax.text(7.6, 2.5, 'freq, real_imp,', fontsize=9, ha='center', color='white')
    ax.text(7.6, 2.1, 'imag_imp', fontsize=9, ha='center', color='white')
    
    # Step 4: Comparator
    ax.add_patch(FancyBboxPatch((9.6, 1.5), 2.2, 2, boxstyle="round,pad=0.1",
                                 facecolor=ORANGE, edgecolor=DARK, linewidth=2))
    ax.text(10.7, 3.1, 'Comparator', fontsize=11, fontweight='bold', ha='center', color='white')
    ax.text(10.7, 2.5, 'MSE, RMSE, L1', fontsize=9, ha='center', color='white')
    ax.text(10.7, 2.0, 'compare()', fontsize=8, ha='center', color='white', family='monospace')
    
    # Arrows
    for x in [2.7, 5.6, 8.9]:
        ax.annotate('', xy=(x + 0.6, 2.5), xytext=(x, 2.5), 
                   arrowprops=dict(arrowstyle='->', color=DARK, lw=2))
    
    # Bottom: data flow
    ax.text(1.6, 0.9, 'Many instances', fontsize=8, ha='center', color=DARK, style='italic')
    ax.text(4.5, 0.9, 'Many signatures', fontsize=8, ha='center', color=DARK, style='italic')
    ax.text(7.6, 0.9, '.npy file data', fontsize=8, ha='center', color=DARK, style='italic')
    ax.text(10.7, 0.9, 'Similarity score', fontsize=8, ha='center', color=DARK, style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'concept_diagram.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated concept_diagram.png")


def generate_signature_comparison():
    """
    Shows actual signature data structure:
    - freq (column 0)
    - real_imp (column 1) 
    - imag_imp (column 2)
    
    Based on PiezoelectricSignature.load_data() in utils.py
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    np.random.seed(42)
    # Simulate realistic impedance data (freq range from __init__.py: 10kHz-150kHz)
    freq = np.linspace(10000, 150000, 500)
    
    # Simulate real impedance with resonance peak
    real_imp_base = 800 + 600 * np.exp(-((freq - 60000) / 25000) ** 2)
    real_imp_1 = real_imp_base + np.random.normal(0, 15, len(freq))
    real_imp_2 = real_imp_base + np.random.normal(0, 15, len(freq))
    
    # Simulate imaginary impedance 
    imag_imp_base = -200 + 400 * np.exp(-((freq - 70000) / 30000) ** 2)
    imag_imp_1 = imag_imp_base + np.random.normal(0, 10, len(freq))
    imag_imp_2 = imag_imp_base + np.random.normal(0, 10, len(freq))
    
    # Plot 1: Real Impedance
    ax1 = axes[0]
    ax1.plot(freq/1000, real_imp_1, color=BLUE, alpha=0.8, label='Signature 1')
    ax1.plot(freq/1000, real_imp_2, color=PURPLE, alpha=0.8, label='Signature 2')
    ax1.set_xlabel('Frequency (kHz)', fontsize=10)
    ax1.set_ylabel('Real Impedance (Ohms)', fontsize=10)
    ax1.set_title('real_imp (Column 1)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor(LIGHT)
    
    # Plot 2: Imaginary Impedance
    ax2 = axes[1]
    ax2.plot(freq/1000, imag_imp_1, color=BLUE, alpha=0.8, label='Signature 1')
    ax2.plot(freq/1000, imag_imp_2, color=PURPLE, alpha=0.8, label='Signature 2')
    ax2.set_xlabel('Frequency (kHz)', fontsize=10)
    ax2.set_ylabel('Imaginary Impedance (Ohms)', fontsize=10)
    ax2.set_title('imag_imp (Column 2)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor(LIGHT)
    
    plt.suptitle('PiezoelectricSignature Data: Same Instance, Different Measurements', 
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'signature_comparison.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated signature_comparison.png")


def generate_methodology_diagram():
    """
    Shows the statistical methodology from run_experiment_1.py:
    1. estimate_normal_dist() -> MultivariateNormalDistribution(mean, cov)
    2. probability_of_multivariant_point() using Mahalanobis distance
    3. estimate_overlap_of_set_with_sample_signals() -> collision rate
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    ax.text(6, 6.5, 'Statistical Methodology (from experiments/run_experiment_1.py)', 
            fontsize=13, fontweight='bold', ha='center', color=DARK)
    
    # Box 1: PDF Estimation
    ax.add_patch(FancyBboxPatch((0.3, 3.8), 3.5, 2.2, boxstyle="round,pad=0.1",
                                 facecolor=LIGHT, edgecolor=BLUE, linewidth=2))
    ax.text(2.05, 5.6, 'estimate_normal_dist()', fontsize=10, fontweight='bold', 
            ha='center', color=BLUE, family='monospace')
    ax.text(2.05, 5.0, 'Returns:', fontsize=9, ha='center', color=DARK)
    ax.text(2.05, 4.5, 'MultivariateNormalDistribution', fontsize=9, ha='center', 
            color=DARK, family='monospace')
    ax.text(2.05, 4.1, '(mean, cov)', fontsize=9, ha='center', color=DARK)
    
    # Box 2: Point Probability  
    ax.add_patch(FancyBboxPatch((4.25, 3.8), 3.5, 2.2, boxstyle="round,pad=0.1",
                                 facecolor=LIGHT, edgecolor=PURPLE, linewidth=2))
    ax.text(6, 5.6, 'probability_of_', fontsize=10, fontweight='bold', 
            ha='center', color=PURPLE, family='monospace')
    ax.text(6, 5.2, 'multivariant_point()', fontsize=10, fontweight='bold', 
            ha='center', color=PURPLE, family='monospace')
    ax.text(6, 4.5, 'Mahalanobis distance', fontsize=9, ha='center', color=DARK)
    ax.text(6, 4.1, '+ chi2 CDF', fontsize=9, ha='center', color=DARK)
    
    # Box 3: Collision Estimation
    ax.add_patch(FancyBboxPatch((8.2, 3.8), 3.5, 2.2, boxstyle="round,pad=0.1",
                                 facecolor=LIGHT, edgecolor=GREEN, linewidth=2))
    ax.text(9.95, 5.6, 'estimate_overlap_of_', fontsize=10, fontweight='bold', 
            ha='center', color=GREEN, family='monospace')
    ax.text(9.95, 5.2, 'set_with_sample_signals()', fontsize=9, fontweight='bold', 
            ha='center', color=GREEN, family='monospace')
    ax.text(9.95, 4.5, 'Monte Carlo sampling', fontsize=9, ha='center', color=DARK)
    ax.text(9.95, 4.1, 'Returns: collision rate', fontsize=9, ha='center', color=DARK)
    
    # Arrows
    ax.annotate('', xy=(4.25, 4.9), xytext=(3.8, 4.9), 
               arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    ax.annotate('', xy=(8.2, 4.9), xytext=(7.75, 4.9), 
               arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    
    # Formula box
    ax.add_patch(FancyBboxPatch((1.5, 1), 9, 2.2, boxstyle="round,pad=0.1",
                                 facecolor='#FEF3C7', edgecolor=ORANGE, linewidth=2))
    ax.text(6, 2.8, 'Collision Rate Calculation', fontsize=11, fontweight='bold', 
            ha='center', color=ORANGE)
    ax.text(6, 2.2, 'collision_rate = sum(collisions) / (num_samples * num_parts)', 
            fontsize=10, ha='center', color=DARK, family='monospace')
    ax.text(6, 1.5, 'where collision = max(num_parts_matched - 1, 0)', 
            fontsize=9, ha='center', color=DARK, style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'methodology_diagram.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated methodology_diagram.png")


def generate_data_structure_diagram():
    """
    Shows actual class hierarchy from utils.py:
    Part -> PartInstance -> PiezoelectricSignature
    
    Based on ALL_PART_TYPES from __init__.py: CON, CONLID, LID, SEN, TUBE
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    ax.text(5, 6.5, 'Class Hierarchy (utils.py)', fontsize=14, fontweight='bold', 
            ha='center', color=DARK)
    
    # Part level
    ax.add_patch(FancyBboxPatch((3, 5), 4, 1, boxstyle="round,pad=0.1",
                                 facecolor=BLUE, edgecolor=DARK, linewidth=2))
    ax.text(5, 5.5, 'Part(part_type)', fontsize=11, ha='center', 
            color='white', fontweight='bold')
    ax.text(5, 5.15, 'CON | CONLID | LID | SEN | TUBE', fontsize=8, ha='center', color='white')
    
    # PartInstance level
    for i, (x, label) in enumerate([(1.5, 'x1'), (4, 'x2'), (6.5, 'x3')]):
        ax.add_patch(FancyBboxPatch((x, 3), 2, 1, boxstyle="round,pad=0.1",
                                     facecolor=PURPLE, edgecolor=DARK, linewidth=2))
        ax.text(x + 1, 3.6, 'PartInstance', fontsize=9, ha='center', color='white', fontweight='bold')
        ax.text(x + 1, 3.2, f'instance_id="{label}"', fontsize=8, ha='center', color='white')
        # Arrow from Part
        ax.annotate('', xy=(x + 1, 4), xytext=(5, 5), 
                   arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    
    # PiezoelectricSignature level
    positions = [(0.5, 'sig_1'), (2.5, 'sig_2'), (4.5, 'sig_1'), (6.5, 'sig_1'), (8.5, 'sig_2')]
    parent_x = [2.5, 2.5, 5, 7.5, 7.5]
    for (x, label), px in zip(positions, parent_x):
        ax.add_patch(FancyBboxPatch((x, 1), 1.5, 1, boxstyle="round,pad=0.1",
                                     facecolor=GREEN, edgecolor=DARK, linewidth=2))
        ax.text(x + 0.75, 1.6, 'Signature', fontsize=8, ha='center', color='white', fontweight='bold')
        ax.text(x + 0.75, 1.25, label, fontsize=7, ha='center', color='white')
        ax.annotate('', xy=(x + 0.75, 2), xytext=(px, 3), 
                   arrowprops=dict(arrowstyle='->', color=DARK, lw=1))
    
    # Legend box
    ax.add_patch(FancyBboxPatch((0.3, 0), 9.4, 0.7, boxstyle="round,pad=0.05",
                                 facecolor=LIGHT, edgecolor=DARK, linewidth=1))
    ax.text(5, 0.35, 'Each Signature: .npy file with [freq, real_imp, imag_imp] arrays', 
            fontsize=9, ha='center', color=DARK, family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'data_structure.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated data_structure.png")


def main():
    print(f"Generating images in {IMAGES_DIR}/\n")
    
    generate_concept_diagram()
    generate_signature_comparison()
    generate_methodology_diagram()
    generate_data_structure_diagram()
    
    print(f"\nDone! Generated {len(os.listdir(IMAGES_DIR))} images.")


if __name__ == '__main__':
    main()
