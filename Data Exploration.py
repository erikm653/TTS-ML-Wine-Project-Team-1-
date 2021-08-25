
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



wine_red = pd.read_csv('winequality-red.csv',sep=';')
wine_white = pd.read_csv('winequality-white.csv',sep=';')

# RED WINE DATA

wine_red.insert(6,'bound sulfur dioxide', wine_red['total sulfur dioxide'] - wine_red['free sulfur dioxide'])


wine_red_qualgroup = wine_red.groupby('quality')

wine_red_descr = wine_red_qualgroup.agg(['mean']).round(2).T#, 'std','sem']).round(2).T

wine_red_sugar_alc = (
                      wine_red.loc[:,['residual sugar','alcohol','quality']]
                          .groupby('quality')
                          .agg([np.mean,np.min,np.max])
                          .round(2)
                          .T
                      )
#print(wine_red_descr)


# WHITE WINE DATA

wine_white.insert(6,'bound sulfur dioxide', wine_white['total sulfur dioxide'] - wine_white['free sulfur dioxide'])

wine_white_qualgroup = wine_white.groupby('quality')

wine_white_descr = wine_white_qualgroup.agg(['mean']).round(2).T#, 'std','sem']).round(2).T

wine_white_sugar_alc = (
                      wine_white.loc[:,['residual sugar','alcohol','quality']]
                          .groupby('quality')
                          .agg([np.mean,np.min,np.max])
                          .round(2)
                          .T
                      )
# WHAT I AM NOTICING:
    #slight monotonic behavior with SULPHATES,CITRIC ACID,VOLATILE ACID
    #variation within total sulfur
    #I see a need for weighted probabilities based on intervals of factor values

#testing
#test#2
