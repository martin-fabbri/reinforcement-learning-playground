import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# implementing random selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward += reward

plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each as was selected")
plt.show()

print(f"Total Reward: {total_reward}")
