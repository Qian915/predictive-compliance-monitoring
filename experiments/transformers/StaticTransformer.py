from sklearn.base import TransformerMixin
import pandas as pd
from time import time

class StaticTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
    
    
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X, y=None):
        start = time()
        
        dt_first = X.groupby(self.case_id_col).first()
        
        # transform numeric cols
        dt_transformed = dt_first[self.num_cols]
        
        # transform cat cols
        if len(self.cat_cols) > 0:
            dt_cat = pd.get_dummies(dt_first[self.cat_cols])
            dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)

        # fill NA with 0 if requested
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
            
        # add missing columns if necessary
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0, index=dt_transformed.index, columns=missing_cols)  # create a df with missing columns
                dt_transformed = pd.concat([dt_transformed, missing_df], axis=1)    # join the df with all missing columns at once
            '''
            for col in missing_cols:
                dt_transformed[col] = 0
            '''
            dt_transformed = dt_transformed[self.columns]
        else:
            self.columns = dt_transformed.columns
        
        self.transform_time = time() - start
        return dt_transformed
    
    def get_feature_names(self):
        return self.columns