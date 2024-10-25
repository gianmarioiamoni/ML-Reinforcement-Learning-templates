# UPPER CONFIDENCE BOUND
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#
# The data set if a simulation of a real time activity where
# to different users is shown randomly an adv choosen in a set of 10
# and we record with 0/1 if the user wouldn't or would click on the showed AD
#
# The 0 and 1 in the dataset are in fact the rewards
#
# For reinforcement learning we don't have to create a matrix of features
# or dependant variables
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing the UCB
#
# 1. at each round (user) n, we consider 2 numbers for each ADi:
#   - Ni(n) = the number of times the ADi was selected up to round n
#   - Ri(n) = the sum of rewards of the ADVi up to round n
import math
N = 10000 # total number of users (rounds)
d = 10 # number of ADs
ads_selected = [] # list of selected ADs at each round
numbers_of_selections = [0] * d # Ni(n) - initialised as a list of ten 0
sums_of_rewards = [0] * d # Ri(n)
total_reward = 0 # the sum of all the rewards received over the round

# 2. From these 2 numbers we compute:
#   - the average reward of ADi up to round n:
#     r''i(n) = Ri(n)/Ni(n)
#
#   - the confidence interval [r''i(n) - DELTAi(n), r''i(n)+DELTAi(n)],where:
#     DELTAi(n) = SQRT(3log(n)/2Ni(n))
#
# 3. We select the ADi that has the maximum UCB r''(n)+DELTAi(n)
for n in range(0, N):
  # per each round:
  # we select an AD, starting from the first
  ad = 0 
  max_upper_bound = 0 # store the max UCB of each round
  # loop over the different ADs, to compare the maximum Upper Confidence Bound
  # by comparing the UCB of each of the ADs
  for i in range(0, d):
    # per each AD
    if numbers_of_selections[i] > 0:
      # if the AD has been selected at least once
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]
      # n start from 0 in the loop, so we use n+1 in the log
      delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
      upper_bound = average_reward + delta_i # maximum UCB
    else:
      # if the AD has not been selected at least once
      # we have to select it
      upper_bound = 1e400 # trick to select all the AD (simulate infinity)
    if (upper_bound > max_upper_bound):
      # update the max_upper_bound
      max_upper_bound = upper_bound
      # select the AD with max uper bound
      ad = i
  # update the variables
  ads_selected.append(ad) # add the selected AD to the list of selected ADs
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset.values[n, ad] # get the reward from the dataset
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward # update the total reward at the round n

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()