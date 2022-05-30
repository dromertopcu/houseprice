import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.base import TransformerMixin
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import joblib
from sklearn.pipeline import Pipeline
import streamlit as st

st.title("H O Ş G E L D İ N İ Z !")

st.title("Lütfen istediğiniz evin özelliklerini seçiniz!")

# %%

#CAT_OPTIONS
HouseStyle_options = ['','One story','One and one-half story','One and one-half story:Unfinished','Two story','Two and one-half story','Two and one-half story:Unfinished','Split Foyer','Split Level']
HouseStyle_dict = {'':"",'One story':'1Story','One and one-half story':'1.5Fin','One and one-half story:Unfinished':'1.5Unf','Two story':'2Story','Two and one-half story':'2.5Fin','Two and one-half story':'2.5Unf','Split Foyer':'SFoyer','Split Level':'SLvl'}
HouseStyle = st.sidebar.selectbox('Style of dwelling', options=HouseStyle_options)

MSZoning_options = ['','Commercial', 'Floating Village Residential','Residential High Density','Residential Low Density','Residential Medium Density']
MSZoning_dict = {'':'','Commercial':'C (all)', 'Floating Village Residential':'FV','Residential High Density':'RH','Residential Low Density':'RL','Residential Medium Density':'RM'}
MSZoning = st.sidebar.selectbox('General Zoning Classification',options=MSZoning_options)

BldgType_options = ['','Single-family Detached',"Two-family Conversion","Duplex","Townhouse End Unit","Townhouse Inside Unit"]
BldgType_dict = {'':'','Single-family Detached':'1Fam',"Two-family Conversion":'2fmCon',"Duplex":'Duplex',"Townhouse End Unit":'TwnhsE',"Townhouse Inside Unit":'Twnhs'}
BldgType = st.sidebar.selectbox('Type of Dwelling',options=BldgType_options)

SaleType_options = ['',"Warranty Deed - Conventional","Warranty Deed - Cash",'Home just constructed and sold','Court Officer Deed/Estate','Contract 15% Down payment regular terms','Contract Low Down payment and low interest','Contract Low Interest','Contract Low Down','Other']
SaleType_dict = {'':'',"Warranty Deed - Conventional":'WD',"Warranty Deed - Cash":'CWD','Home just constructed and sold':'New','Court Officer Deed/Estate':'COD','Contract 15% Down payment regular terms':'Con','Contract Low Down payment and low interest':'ConLw','Contract Low Interest':'ConLI','Contract Low Down':'ConLD','Other':'Oth'}
SaleType = st.sidebar.selectbox('Type of Sale',options=SaleType_options)

Heating_options = ['','Floor Furnace','Gas forced warm air furnace', 'Gas hot water', 'Gravity furnace', 'other than gas','Wall furnace']
Heating_dict = {'':'','Floor Furnace':'Floor','Gas forced warm air furnace':'GasA', 'Gas hot water':'GasW', 'Gravity furnace':'Grav', 'other than gas':'OthW','Wall furnace':'Wall'}
Heating = st.sidebar.selectbox('Type of Heating',options=Heating_options)

#NUM_OPTIONS
OverallQual = st.sidebar.select_slider(
     'Rates the overall material and finish of the house',
     options=[*range(1,11)])
GrLivArea = st.sidebar.select_slider(
     'Above grade (ground) living area square feet',
     options=[*range(334,5643)])
GarageArea = st.sidebar.select_slider(
     'Size of garage in square feet',
     options=[*range(0,1419)])
TotalBsmtSF = st.sidebar.select_slider(
     'Total square feet of basement area',
     options=[*range(0,6111)])
FullBath = st.sidebar.select_slider(
     'Full bathrooms above grade',
     options=[*range(0,4)])


soru_list1 = [HouseStyle_dict,MSZoning_dict,BldgType_dict,SaleType_dict,Heating_dict]
soru_list2 = [HouseStyle,MSZoning,BldgType,SaleType,Heating]
soru_list3 = []
for i,j in zip(soru_list2,soru_list1) :
    soru_list3.append(j[i])
soru_list3 += [OverallQual,GrLivArea,GarageArea,TotalBsmtSF,FullBath]

soru_list4 = ['HouseStyle','MSZoning','BldgType','SaleType','Heating','OverallQual','GrLivArea','GarageArea','TotalBsmtSF','FullBath']

# %% 

single_row = pd.read_csv('single_row.csv',index_col =0)
plus = pd.read_csv('plus.csv',index_col =0)


class SparseMatrix(TransformerMixin):
    def __init__(self):
        None
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        categorical_columns = ['MSSubClass','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'Electrical', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish','GarageQual','GarageCond', 'PavedDrive', 'Fence', 'SaleType', 'SaleCondition']
        X[categorical_columns] = X[categorical_columns].astype(str)
        print(X[categorical_columns].shape)
        ohe = joblib.load('ohe.joblib')
        hot = ohe.transform(X[categorical_columns].astype(str))
        print(hot.getcol)
        cold_df = X.select_dtypes(exclude=["object"])
        print(cold_df.shape)
        cold = csr_matrix(cold_df.values)
        print(cold_df.info())
        print(cold.getcol)
        final_sparse_matrix = hstack((hot, cold))
        final_csr_matrix = final_sparse_matrix.tocsr()
        return final_csr_matrix

data_pipeline = Pipeline([('sparse', SparseMatrix())])

bst = xgb.Booster()

bst.load_model('housepricexgb.model')

if st.sidebar.button('Show House Price'):
    for i in range(len(soru_list3)):
        if soru_list3[i] != '' :
            single_row.loc[0,soru_list4[i]] = soru_list3[i]
    single_row_plus = pd.concat([single_row,plus])
    single_row_plus_transformed = data_pipeline.fit_transform(single_row_plus)
    xgmat = xgb.DMatrix(single_row_plus_transformed,missing = -999.0)
    ypred = bst.predict(xgmat)
    st.title('Seçtiğiniz evin tahmini fiyatı : ')
    st.title(np.round(ypred[0]))
