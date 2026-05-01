#!/usr/bin/env python3
import json

# Read the notebook
with open('student_performance_ml.ipynb', 'r') as f:
    notebook = json.load(f)

# New markdown cell
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["# Accuracy Comparison - Bar Chart of All Classifiers"]
}

# New code cell for accuracy comparison
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "# Extract accuracy values for all classifiers\n",
        "classifiers = list(results.keys())\n",
        "accuracies = [results[name]['accuracy'] for name in classifiers]\n",
        "\n",
        "# Create bar chart\n",
        "fig, ax = plt.subplots(figsize=(12, 7))\n",
        "\n",
        "# Define colors\n",
        "colors = ['#7c3aed', '#22d3ee', '#22c55e', '#f59e0b', '#ef4444']\n",
        "bars = ax.bar(classifiers, accuracies, color=colors, edgecolor='white', linewidth=2.5, alpha=0.8)\n",
        "\n",
        "# Add value labels on top of bars\n",
        "for bar, accuracy in zip(bars, accuracies):\n",
        "    height = bar.get_height()\n",
        "    ax.text(bar.get_x() + bar.get_width()/2., height,\n",
        "            f'{accuracy:.4f}',\n",
        "            ha='center', va='bottom', fontsize=12, fontweight='bold')\n",
        "\n",
        "# Customize the plot\n",
        "ax.set_ylim([0, 1.0])\n",
        "ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')\n",
        "ax.set_xlabel('Machine Learning Classifiers', fontsize=12, fontweight='bold')\n",
        "ax.set_title('Accuracy Comparison of All Five Classifiers', fontsize=14, fontweight='bold', pad=20)\n",
        "ax.grid(axis='y', alpha=0.3, linestyle='--')\n",
        "ax.set_axisbelow(True)\n",
        "\n",
        "# Format y-axis as percentage\n",
        "ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))\n",
        "\n",
        "# Rotate x-axis labels for better readability\n",
        "plt.xticks(rotation=15, ha='right')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('plots/accuracy_comparison.png', dpi=300)\n",
        "plt.show()\n",
        "\n",
        "# Print accuracy summary table\n",
        "print(\"\\n\" + \"=\" * 70)\n",
        "print(\"ACCURACY COMPARISON - ALL CLASSIFIERS\")\n",
        "print(\"=\" * 70)\n",
        "for classifier, accuracy in zip(classifiers, accuracies):\n",
        "    print(f\"{classifier:.<35} {accuracy:.4f} ({accuracy*100:.2f}%)\")\n",
        "print(\"=\" * 70)\n",
        "print(f\"Best Performing Model: {best_model_name} with {results[best_model_name]['accuracy']:.4f} accuracy\")"
    ]
}

# Add cells to notebook
notebook['cells'].append(markdown_cell)
notebook['cells'].append(code_cell)

# Write back
with open('student_performance_ml.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Accuracy comparison cells added successfully!")
