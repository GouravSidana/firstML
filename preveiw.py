import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your specific dataset
# Ensure 'ML_Ready_Dataset.csv' is in the same folder as your Python script
df = pd.read_csv('data.csv')

# Set the visual style for the plots
sns.set_theme(style="whitegrid")

# Image 1: PCE vs VOC (Points Only)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, 
    x='VOC', 
    y='PCE', 
    alpha=0.7, 
    color='#005088',
    s=50 # 's' controls the size of the dots
)
plt.title('PCE vs. VOC (Open-Circuit Voltage)', fontsize=14)
plt.xlabel('VOC', fontsize=12)
plt.ylabel('PCE (Efficiency)', fontsize=12)
plt.tight_layout()
plt.savefig('pce_vs_voc_scatter.png', dpi=300)
plt.close()


# Image 2: PCE vs FF (Points Only)

plt.figure(figsize=(8, 6)) 
sns.scatterplot(
    data=df, 
    x='FF', 
    y='PCE', 
    alpha=0.7, 
    color='#11caa0',
    s=50
)
plt.title('PCE vs. FF (Fill Factor)', fontsize=14)
plt.xlabel('FF', fontsize=12)
plt.ylabel('PCE (Efficiency)', fontsize=12)
plt.tight_layout()
plt.savefig('pce_vs_ff_scatter.png', dpi=300) 
plt.close()

# Image 3: PCE vs JSC (Points Only)

plt.figure(figsize=(8, 6)) 
sns.scatterplot(
    data=df, 
    x='JSC', 
    y='PCE', 
    alpha=0.7, 
    color='#e67e22',
    s=50
)
plt.title('PCE vs. JSC (Short-Circuit Current)', fontsize=14)
plt.xlabel('JSC', fontsize=12)
plt.ylabel('PCE (Efficiency)', fontsize=12)
plt.tight_layout()
plt.savefig('pce_vs_jsc_scatter.png', dpi=300) 
plt.close()

print("Successfully saved 3 pure scatter plots: 'pce_vs_voc_scatter.png', 'pce_vs_ff_scatter.png', and 'pce_vs_jsc_scatter.png'")