import joblib
from sklearn.metrics import precision_score, recall_score, fbeta_score

def train_model(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def compute_model_metrics(y, preds):
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    fbeta = fbeta_score(y, preds, beta=1)
    return precision, recall, fbeta

def inference(model, X):
    return model.predict(X)

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)

def performance_on_categorical_slice(df, feature, y, preds):
    """
    Computes the model metrics on slices for a given categorical feature.
    
    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the features.
    feature : str
        Feature to compute the slice on.
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    metrics : list
        List of dictionaries containing metrics for each slice.
    """
    import logging
    
    slice_metrics = []
    
    for cls in df[feature].unique():
        df_temp = df[df[feature] == cls]
        y_slice = y[df_temp.index]
        preds_slice = preds[df_temp.index]
        
        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
        
        slice_info = {
            'feature': feature,
            'value': cls,
            'count': len(df_temp),
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta
        }
        
        slice_metrics.append(slice_info)
        
        logging.info(f"{feature}: {cls}, Count: {len(df_temp)}")
        logging.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")
    
    return slice_metrics

def performance_on_categorical_slice(df, feature, y, preds):
    """
    Computes the model metrics on slices for a given categorical feature.
    
    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the features.
    feature : str
        Feature to compute the slice on.
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    metrics : list
        List of dictionaries containing metrics for each slice.
    """
    import logging
    
    slice_metrics = []
    
    for cls in df[feature].unique():
        df_temp = df[df[feature] == cls]
        y_slice = y[df_temp.index]
        preds_slice = preds[df_temp.index]
        
        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
        
        slice_info = {
            'feature': feature,
            'value': cls,
            'count': len(df_temp),
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta
        }
        
        slice_metrics.append(slice_info)
        
        logging.info(f"{feature}: {cls}, Count: {len(df_temp)}")
        logging.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")
    
    return slice_metrics
