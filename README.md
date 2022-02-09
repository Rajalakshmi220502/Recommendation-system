# Recommendation-system
#importing necessary libraries
import pandas as pd
import numpy as np

#reading data
df1 = pd.read_csv(r"C:\Users\welcome\AppData\Local\Temp\Rar$DIa9088.13763\movies.csv")

df1.head()

df2 = pd.read_csv(r"C:\Users\welcome\AppData\Local\Temp\Rar$DIa9088.11347\ratings.csv")

df2.head()

df1.describe

df2.describe

#Merging two data as single data
df = pd.merge(df1,df2,on = "movieId")
df.head()

df.describe()

df.isnull().sum()




#visualization
import matplotlib.pyplot as plt
import seaborn as sns


df.groupby('title')['rating'].mean().sort_values(ascending = False).head()


df.groupby('title')['rating'].count().sort_values(ascending = False).head()

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.tail()

plt.figure(figsize=(10,6))
ratings['num of ratings'].hist(bins = 50)

plt.figure(figsize =(10,6))
ratings['rating'].hist(bins = 50)

sns.jointplot(x='rating',y='num of ratings',data = ratings,alpha = 0.5)

## creating user iteraction matrix


movie_matrix = df.pivot_table(index='userId',columns= 'title',values = 'rating')

movie_matrix

# most rated movies
ratings.sort_values('num of ratings',ascending= False).head(10)

#Removing movies which have less than 10 users who rated it.and fill remaining null with 0
movie_matrix = movie_matrix.dropna(thresh = 5,axis = 1).fillna(0)

movie_matrix

##making recommendation - example movie - pulp fiction

#fetching ratings for the movie
pulpfiction_user_rating = movie_matrix['Pulp Fiction (1994)']

pulpfiction_user_rating

#finding similar movie to that pulpfiction movie using correlation 
similar_to_pulpfiction = movie_matrix.corrwith(pulpfiction_user_rating)


similar_to_pulpfiction.dropna(inplace = True)
similar_to_pulpfiction.head()

corr_pulpfiction = pd.DataFrame(similar_to_pulpfiction,columns=['correlation'])
corr_pulpfiction.dropna(inplace= True)
corr_pulpfiction.head()

 corr_pulpfiction = corr_pulpfiction.join(ratings['num of ratings'])
corr_pulpfiction.head(10)

corr_pulpfiction[corr_pulpfiction['num of ratings'] > 20].sort_values(by='correlation',ascending = False).head(10)

trainset,testset = train_test_split(df,test_size =.20)

from scipy.sparse import csr_matrix



movie_csr_matrix = csr_matrix(movie_matrix.values)

from sklearn.neighbors import NearestNeighbors


model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_csr_matrix)
