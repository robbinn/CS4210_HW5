#-------------------------------------------------------------------------
# AUTHOR: Siwen Wang
# FILENAME: collaborative_filtering
# SPECIFICATION: Assignment #5, question #5
# FOR: CS 4200- Assignment #5
# TIME SPENT: 2 Hours
#-----------------------------------------------------------*/

#importing some Python libraries
import operator
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()
X_training = np.array(df.values)
fit_x_training = []
for data in X_training:
    temp = []
    for number in data:
        if isinstance(number,float):
            temp.append(number)
    fit_x_training.append(temp)

#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
   # do this to calculate the similarity:
   #vec1 = np.array([[1,1,0,1,1]])
   #vec2 = np.array([[0,1,0,1,1]])
   #cosine_similarity(vec1, vec2)
   #do not forget to discard the first column (User ID) when calculating the similarities
   #--> add your Python code here
vec1 = np.array([fit_x_training[99]])
rank = {}
for i in range(len(fit_x_training)-1):
    vec2 = np.array([fit_x_training[i]])
    rank[i] = cosine_similarity(vec1, vec2)[0][0]

sorted_rank = sorted(rank.items(), key=operator.itemgetter(1))
sorted_rank.reverse()

# Average of user #100
avg_target = sum(vec1[0]) / len(vec1[0])

# For Gallery
top = 0
down = 0
# gallery col in X_training data
gallery_index = 1
for i in range(len(sorted_rank)):
    temp = sorted_rank[i][1] * (float(X_training[sorted_rank[i][0]][gallery_index]) - sum(fit_x_training[i])/len(fit_x_training[i]))
    top += temp
    down += sorted_rank[i][1]
result_gallary = avg_target + top/down
print(result_gallary)

# For restaurants
top = 0
down = 0
# restaurants col in X_training data
restaurants = 4
for i in range(len(sorted_rank)):
    temp = sorted_rank[i][1] * (float(X_training[sorted_rank[i][0]][restaurants]) - sum(fit_x_training[i])/len(fit_x_training[i]))
    top += temp
    down += sorted_rank[i][1]
result_restaurants = avg_target + top/down
print(result_restaurants)

# top10Index = []
# for key in sorted(rank)[-10:]:
#     top10Index.append(rank[key])
# print(top10Index)

    # print(key, rank[key])
   #find the top 10 similar users to the active user according to the similarity calculated before
   #--> add your Python code here

   #Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
   #--> add your Python code here



