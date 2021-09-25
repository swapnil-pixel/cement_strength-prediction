import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score

data=pd.read_excel("Concrete_Data.xls")
data = data.rename(columns={"Cement (component 1)(kg in a m^3 mixture)":"cement",
                        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)":"Blast Furnace slag",
                        "Fly Ash (component 3)(kg in a m^3 mixture)":"Fly Ash ",
                        "Water  (component 4)(kg in a m^3 mixture)":"water",
                        "Superplasticizer (component 5)(kg in a m^3 mixture)":"superplastic",
                        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)":"Coarse Aggregate ",
                        "Fine Aggregate (component 7)(kg in a m^3 mixture)":"Fine Aggregate ",
                        "Age (day)":"age",
                        "Concrete compressive strength(MPa, megapascals) ":"strength"})
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data[['cement', 'Blast Furnace slag', 'Fly Ash ', 'water', 'superplastic',
       'Coarse Aggregate ', 'Fine Aggregate ', 'age']]=sc.fit_transform(data[['cement', 'Blast Furnace slag', 'Fly Ash ', 'water', 'superplastic',
       'Coarse Aggregate ', 'Fine Aggregate ', 'age']])
x=data.drop('strength',axis=1)
y=data['strength']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, train_size=0.7, test_size=0.3, random_state=100)
RF=RandomForestRegressor(n_estimators=100,min_samples_split= 2,min_samples_leaf=1,max_features= 'sqrt',max_depth=20,bootstrap=False,random_state=42)
RF.fit(x_train,y_train)
y_train_pred=RF.predict(x_train)
y_test_pred=RF.predict(x_test)
#print(y_test_pred)
Rsqr_test=round(r2_score(y_test,y_test_pred)*100,2)
    
print("Rsqr is ",Rsqr_test)
import pickle
# # Saving model to disk
pickle.dump(RF, open('RF.pkl','wb'))
model=pickle.load(open('RF.pkl','rb'))