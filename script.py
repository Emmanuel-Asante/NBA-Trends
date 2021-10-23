# Import modules
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

import codecademylib3
np.set_printoptions(suppress=True, precision = 2)

# Read csv file into a variable called nba
nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

# Output the first 10 rows of nba_2010, nba_2014
print(nba_2010.head())
print(nba_2014.head())

# Create a series named knicks_pts from the dataframe nba_2010 where fran_id = "Knicks"
knicks_pts = nba_2010.pts[nba.fran_id == "Knicks"]

# Create a series named nets_pts from the dataframe nba_2010 where fran_id = "Nets"
nets_pts = nba_2010.pts[nba.fran_id == "Nets"]

# Calculate the mean of knicks_pts and assign the result to a variable named knicks_pts_mean
knicks_pts_mean = np.mean(knicks_pts)

# Calculate the mean of nets_pts and assign the result to a variable named nets_pts_mean
nets_pts_mean = np.mean(nets_pts)

# Calculate the difference in means for knicks_pts_mean and nets_pts_mean. Assign the result to a variable named diff_means_2010
diff_means_2010 = knicks_pts_mean - nets_pts_mean

# Output mean_diff
print(diff_means_2010)

# Do you think fran_id and pts are associated? Why or why not?
print("From the dataset nba_2010, the mean difference of " + str(diff_means_2010) + " indicate that there is an association between fran_id and pts\n")

# Create a set of overlapping histograms that can be used to compare knicks_pts and nets_pts
plt.hist(knicks_pts, alpha=0.8, normed = True, label="knicks")
plt.hist(nets_pts, alpha=0.8, normed = True, label="nets")
plt.legend()
plt.show()

# Create a series named knicks_pts_2014 from the dataframe nba_2014 where fran_id = "Knicks"
knicks_pts_2014 = nba_2014.pts[nba.fran_id == "Knicks"]

# Create a series named nets_pts_2014 from the dataframe nba_2014 where fran_id = "Nets"
nets_pts_2014 = nba_2014.pts[nba.fran_id == "Nets"]

# Calculate the mean of knicks_pts_2014 and assign the result to a variable named knicks_pts_mean_2014
knicks_pts_mean_2014 = np.mean(knicks_pts_2014)

# Calculate the mean of nets_pts_2014 and assign the result to a variable named nets_pts_mean_2014
nets_pts_mean_2014 = np.mean(nets_pts_2014)

# Calculate the difference in means for knicks_pts_mean_2014 and nets_pts_mean_2014. Assign the result to a variable named diff_means_2014
diff_means_2014 = knicks_pts_mean_2014 - nets_pts_mean_2014

# Output mean_diff
print(diff_means_2014)

# Do you think fran_id and pts are associated? Why or why not?
print("From nba_2014 dataset, the mean difference of " + str(diff_means_2014) + " indicate that there is no clear association between fran_id and pts")

# Create a set of overlapping histograms that can be used to compare knicks_pts_2014 and nets_pts_2014
plt.clf()
plt.hist(knicks_pts_2014, alpha=0.8, normed = True, label="knicks")
plt.hist(nets_pts_2014, alpha=0.8, normed = True, label="nets")
plt.legend()
plt.show()

# Using nba_2010, generate side-by-side boxplots with points scored (pts) on the y-axis and team (fran_id) on the x-axis.
plt.clf()
sns.boxplot(data = nba_2010, x = "fran_id", y = "pts")
plt.show()

# Is there any overlap between the boxes? Does this chart suggest that fran_id and pts are associated? Which pairs of teams, if any, earn different average scores per game?
print("\nFrom the boxplots for nba_2010, There are clear overlaps between the boxes, these indicate some level of associations between fran_id and pts. Pairs of teams that earn different average scores per game are\n1. Celtics and Knicks\n2. Celtics and Nets\n3. Knicks and Nets\n4. Nets and Thunder\n5. Nets and Spurs\n")

# Calculate a table of frequencies that shows the counts of game_result and game_location and save the result as location_result_freq
location_result_freq = pd.crosstab(nba_2010.game_result, nba_2010.game_location)

# Print out location_result_freq
print(location_result_freq)

# Convert location_result_freq to a table of proportions and save the result as location_result_proportions
location_result_proportions = location_result_freq / len(nba_2010)

# Display a new line
print("\n")

# Output location_result_proportions
print(location_result_proportions)

# Display a new line
print("\n")

# Calculate the expected contingency table and the Chi-Square statistic of location_result_freq and save your results as expected, chi2
chi2, pval, dof, expected = chi2_contingency(location_result_freq)

# Output the expected contingency table
print(expected)

# Display a new line
print("\n")

# Output the Chi-Square statistic
print(chi2)

# Does the actual contingency table look similar to the expected table — or different? Based on this output, do you think there is an association between these variables?
print("\nThe actual contingency table looks different as compare to the expected table.\n\nA Chi-Square statistic of " + str(chi2) + " indicates a strong association  between the variables (game_result and game_location).\n")

# Using nba_2010, calculate the covariance between forecast (538’s projected win probability) and point_diff (the margin of victory/defeat) in the dataset. Save the result as find_cov_forecast_point_diff
find_cov_forecast_point_diff = np.cov(nba_2010.forecast, nba_2010.point_diff)

# Print out find_cov_forecast_point_diff
print(find_cov_forecast_point_diff)

# What is the covariance between these two variables?
print("\nThe covariance between the variables forecast and point_diff of nba_2010 is 1.37\n")

# Using nba_2010, calculate the correlation between forecast and point_diff, and save the result as corr_forecast_point_diff
corr_forecast_point_diff, p = pearsonr(nba_2010.forecast, nba_2010.point_diff)

# Output corr_forecast_point_diff
print(corr_forecast_point_diff)

# Does this value suggest an association between the two variables?
print("\nThe calculated correlation between forecast and point_diff of nba_2010 i.e {}, shows that there is a linear association between the two variables.".format(corr_forecast_point_diff))

# Generate a scatter plot of forecast (on the x-axis) and point_diff (on the y-axis)
plt.clf()
plt.scatter(x = nba_2010.forecast, y = nba_2010.point_diff)
plt.xlabel("Forecasted Win Prob.")
plt.ylabel("Point Differential")
plt.show()