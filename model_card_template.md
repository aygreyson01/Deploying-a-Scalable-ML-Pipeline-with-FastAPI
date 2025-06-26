# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a **RandomForestClassifier** trained to predict whether an individual's annual income exceeds $50,000 based on demographic and socioeconomic features from the 1994 Census dataset. The model was developed as part of a machine learning pipeline deployment project using FastAPI.

**Model Architecture:**
- Algorithm: Random Forest Classifier
- Number of estimators: 100
- Max depth: None (trees grow until leaves are pure)
- Random state: 42 (for reproducibility)
- Training framework: scikit-learn

**Model Version:** 1.0  
**Date:** June 2025  
**Developers:** ML Pipeline Project Team

## Intended Use

**Primary Use Cases:**
- Educational demonstration of ML pipeline deployment
- Research into income prediction based on demographic factors
- Understanding socioeconomic patterns in 1994 Census data

**Intended Users:**
- Data scientists and ML engineers learning deployment practices
- Researchers studying income inequality and demographic factors
- Students learning classification techniques

**Out-of-Scope Uses:**
- Real-world hiring or employment decisions
- Financial lending or credit decisions  
- Any high-stakes decision making affecting individuals
- Modern income prediction (data is from 1994)

## Training Data

**Dataset:** Adult Census Income Dataset (UCI Machine Learning Repository)  
**Source:** 1994 Census Bureau database  
**Size:** 32,561 total samples  
**Training Split:** 26,048 samples (80%)

**Features (15 total):**
- **Demographic:** age, sex, race, native-country
- **Work-related:** workclass, occupation, hours-per-week
- **Education:** education, education-num  
- **Family:** marital-status, relationship
- **Financial:** capital-gain, capital-loss, fnlgt (final weight)

**Target Variable:** Binary classification of salary (≤50K vs >50K)

**Data Preprocessing:**
- Missing values ('?' symbols) replaced with most frequent values:
  - workclass: '?' → 'Private' 
  - occupation: '?' → 'Prof-specialty'
  - native-country: '?' → 'United-States'
- Categorical features one-hot encoded (105 total dimensions)
- Target labels binary encoded (0: ≤50K, 1: >50K)
- Stratified train/test split to maintain class distribution

## Evaluation Data

**Test Set:** 6,513 samples (20% of total data)  
**Split Method:** Stratified random split with random_state=42  
**Class Distribution:** Maintained original distribution (≤50K: ~76%, >50K: ~24%)

The test set contains the same feature distributions as the training set and was not used during model development or hyperparameter tuning.

## Metrics

**Primary Metrics:** Precision, Recall, and F1-Score for binary classification

**Overall Model Performance:**
- **Precision:** 0.7335 (73.35%)
- **Recall:** 0.6250 (62.50%)  
- **F1-Score:** 0.6749 (67.49%)

**Interpretation:**
- **Precision:** Of individuals predicted to earn >50K, 73.35% actually do
- **Recall:** The model correctly identifies 62.50% of individuals who earn >50K
- **F1-Score:** Balanced measure showing overall classification performance of 67.49%

**Performance Variation Across Groups:**
Performance varies significantly across demographic groups (see slice_output.txt for detailed breakdowns):

*Notable patterns:*
- **Education:** Higher performance for advanced degrees (Doctorate: 86.18% F1, Masters: 83.38% F1)
- **Occupation:** Strong performance for professional roles (Prof-specialty: 77.22% F1, Exec-managerial: 77.79% F1)
- **Relationship:** Married individuals show better prediction accuracy
- **Small sample bias:** Some groups with very few samples show perfect scores (likely overfitting)

## Ethical Considerations

**Bias and Fairness Concerns:**
- **Historical Bias:** Model trained on 1994 data reflects historical inequalities and social structures
- **Demographic Disparities:** Performance varies across racial, gender, and geographic groups
- **Representation Issues:** Some minority groups have very small sample sizes, leading to unreliable predictions

**Potential Harms:**
- Could perpetuate historical biases if used for decision-making
- May underperform for underrepresented demographic groups
- Risk of discrimination if deployed without careful bias assessment

**Mitigation Strategies:**
- Comprehensive slice-based evaluation across all demographic groups
- Documentation of performance disparities  
- Clear guidance against use in high-stakes decisions
- Regular bias auditing if model were to be deployed

## Caveats and Recommendations

**Limitations:**
1. **Data Age:** Training data from 1994 may not reflect current economic conditions or social structures
2. **Missing Data Handling:** Simple imputation strategy may introduce bias
3. **Feature Engineering:** Basic one-hot encoding may not capture complex relationships
4. **Class Imbalance:** 76% vs 24% distribution may bias toward majority class
5. **Small Sample Bias:** Some demographic groups have insufficient data for reliable evaluation

**Recommendations for Use:**
1. **Educational/Research Only:** Do not use for real-world individual decisions
2. **Bias Assessment:** Conduct thorough fairness evaluation before any deployment
3. **Data Updates:** Retrain with modern data if contemporary predictions are needed
4. **Ensemble Methods:** Consider combining with other models for better performance
5. **Continuous Monitoring:** Implement ongoing bias and performance monitoring

**Technical Improvements:**
- Address class imbalance with sampling techniques or cost-sensitive learning
- Implement more sophisticated missing data handling
- Explore feature engineering and selection techniques
- Consider ensemble methods or neural networks for performance gains
- Add fairness constraints during training

**Deployment Considerations:**
- Implement comprehensive logging and monitoring
- Establish clear model governance and update procedures  
- Create detailed API documentation and usage guidelines
- Set up automated retraining pipelines for data drift detection
