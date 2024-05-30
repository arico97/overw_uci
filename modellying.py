import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
import joblib



mapping = {'Normal_Weight':'NW', 'Overweight_Level_I':'OW1', 'Overweight_Level_II':'OW2',
       'Obesity_Type_I':'O1', 'Insufficient_Weight':'IW', 'Obesity_Type_II':'O2',
       'Obesity_Type_III':'O3'
}
inv_map = {v: k for k, v in mapping.items()}
columns = ['Gender',
 'Age', 
 'family_history_with_overweight',
 'FAVC',
 'FCVC',
 'NCP',
 'CAEC',
 'SMOKE',
 'CH2O',
 'SCC',
 'FAF',
 'TUE',
 'CALC',
 'MTRANS']

'''
try:
    oe = joblib.load('./model/oe.joblib') 
    le = joblib.load('./model/le.joblib') 
except OSError as e:
    oe = OrdinalEncoder()
    le = LabelEncoder()
'''
oe = OrdinalEncoder()
le = LabelEncoder()    

class patient_pred:
    def __init__(self, features):
        self.features = features
        self.NObeyesdad = None
 
    def read_data(self,path):
        df=pd.read_csv(path)
        return df

    def preprocess_data(self,df):
        df.NCP=df.NCP.round()
        df.FAF=df.FAF.round()
        df.TUE=df.TUE.round()
        df.rename(columns={'family_history_with_overweight':'FHO'},inplace=True)
        df.replace({'NObeyesdad':mapping},inplace=True)
        df_ct=df[['Gender','SMOKE','FHO',
                            'FAVC','CAEC','SCC',
                            'CALC','MTRANS']]
        Y = df['NObeyesdad'] 
        df_categorized = df.copy()
        df_categorized[['Gender','SMOKE','FHO','FAVC','CAEC','SCC',
                                'CALC','MTRANS']] = oe.fit_transform(df_ct)
        df_categorized['NObeyesdad'] = le.fit_transform(Y)  
      #  joblib.dump(oe,'./model/oe.joblib') 
      #  joblib.dump(le,'./model/le.joblib') 
        return df_categorized
    
    def preprocess_input_pred(self,df):
        df.NCP=df.NCP.round()
        df.FAF=df.FAF.round()
        df.TUE=df.TUE.round()
        df.rename(columns={'family_history_with_overweight':'FHO'},inplace=True)
        df_ct=df[['Gender','SMOKE','FHO', 'FAVC','CAEC','SCC','CALC','MTRANS']] 
        df_categorized = df.copy()

        print(type(df_ct))

        print(oe.categories_)
        df_transf = oe.transform(df_ct) 
        df_categorized[['Gender','SMOKE','FHO','FAVC','CAEC','SCC', 'CALC','MTRANS']] = df_transf
        return df_categorized

    

    def model_data(self,df):
        features_columns=df.columns[~df.columns.isin(['NObeyesdad','Height','Weight'])]
        X,Y = df[features_columns],df['NObeyesdad']
       # X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.1, random_state=2)
        parameters = {'max_depth':[None],'n_estimators':[1000]}
        rf = RandomForestClassifier()
        clf = GridSearchCV(rf, parameters,return_train_score=True)
        print('Time to train data')
        clf.fit(X,Y)
       # joblib.dump(clf, "./model/model.pkl") 
        print('Data trained successfully')
        return clf

    def predict_NObeyesdad(self,path):
        
            df=self.read_data(path)
            df_categ=self.preprocess_data(df)
            model= self.model_data(df_categ) 
            feat = dict(map(lambda i,j: (i,j), columns, self.features))
            f=  pd.DataFrame(feat,index=[0])
            f['NObeyesdad']=None
            features_prep=self.preprocess_input_pred(f) 
            features_prep.drop(columns=['NObeyesdad'],inplace=True)
            prediction=model.predict(features_prep.values) 
            target_decoded = le.inverse_transform(np.array(prediction))
            self.NObeyesdad = inv_map[target_decoded[0]]

'''
    
    def predict_NObeyesdad(self,path):
        try:
            model = joblib.load("./model/model.pkl")
        except:
            model = None
        if model is None:
            df=self.read_data(path)
            df_categ=self.preprocess_data(df)
            self.model_data(df_categ)
            self.predict_NObeyesdad(path)
        else:
            feat = dict(map(lambda i,j: (i,j), columns, self.features))
            f=  pd.DataFrame(feat,index=[0])
            f['NObeyesdad']=None
            features_prep=self.preprocess_input_pred(f) 
            features_prep.drop(columns=['NObeyesdad'],inplace=True)
            prediction=model.predict(features_prep.values) 
            target_decoded = le.inverse_transform(np.array(prediction))
            self.NObeyesdad = inv_map[target_decoded[0]]
        
'''