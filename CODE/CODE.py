import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from xgboost import XGBClassifier

# 1. Load Dataset
file_path = "infection_complete_11.csv"
df = pd.read_csv(file_path)

target_col = "label"

if target_col not in df.columns:
    possible_names = [c for c in df.columns if c.lower() in ['label','lable','target','infection']]
    if possible_names:
        target_col = possible_names[0]
    else:
        raise ValueError("Target column not found")

cols_to_drop = [c for c in ['id','patient_id','Unnamed: 0'] if c in df.columns]

X = df.drop(columns=[target_col] + cols_to_drop)
y = df[target_col].astype(int)

# Encode categorical
X = pd.get_dummies(X, drop_first=True)

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

feature_names = X.columns.tolist()

# Calculate imbalance ratio
pos = sum(y==1)
neg = sum(y==0)
scale_pos_weight = neg/pos

# 2. Model Definitions
models = {

"Logistic Regression": LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
),

"Random Forest": RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
),

"AdaBoost": AdaBoostClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
),

"XGBoost": XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)
}

# 3. Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

plt.figure(figsize=(8,6))

conf_matrices = {}

for model_name, model in models.items():

    metrics = {"acc":[],"sen":[],"spe":[],"f1":[],"auc":[]}

    mean_fpr = np.linspace(0,1,100)
    tprs=[]

    for train_idx, test_idx in skf.split(X_imputed,y):

        X_train,X_test = X_imputed[train_idx],X_imputed[test_idx]
        y_train,y_test = y.iloc[train_idx],y.iloc[test_idx]

        if "Logistic" in model_name:
            scaler=StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.transform(X_test)

        model.fit(X_train,y_train)

        y_pred=model.predict(X_test)
        y_prob=model.predict_proba(X_test)[:,1]

        tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()

        metrics["acc"].append(accuracy_score(y_test,y_pred))
        metrics["sen"].append(recall_score(y_test,y_pred))
        metrics["spe"].append(tn/(tn+fp))
        metrics["f1"].append(f1_score(y_test,y_pred))
        metrics["auc"].append(roc_auc_score(y_test,y_prob))

        fpr,tpr,_=roc_curve(y_test,y_prob)
        tprs.append(np.interp(mean_fpr,fpr,tpr))

        conf_matrices[model_name]=confusion_matrix(y_test,y_pred)

    results.append({
        "Model":model_name,
        "Accuracy":f"{np.mean(metrics['acc']):.3f} ± {np.std(metrics['acc']):.3f}",
        "Sensitivity":f"{np.mean(metrics['sen']):.3f} ± {np.std(metrics['sen']):.3f}",
        "Specificity":f"{np.mean(metrics['spe']):.3f} ± {np.std(metrics['spe']):.3f}",
        "F1-Score":f"{np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}",
        "ROC-AUC":f"{np.mean(metrics['auc']):.3f} ± {np.std(metrics['auc']):.3f}"
    })

    mean_tpr=np.mean(tprs,axis=0)

    plt.plot(mean_fpr,mean_tpr,label=f"{model_name} (AUC={np.mean(metrics['auc']):.3f})")

# ROC Plot
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("roc_curve_models.tiff", dpi=300, format='tiff')
plt.show()

# Results Table
results_df=pd.DataFrame(results)

print("\n============================")
print("MODEL PERFORMANCE")
print("============================")
print(results_df.to_string(index=False))

# Confusion matrices
for model_name,cm in conf_matrices.items():

    plt.figure(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False)

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(f"cm_{model_name}.tiff", dpi=300, format='tiff')

    plt.show()

# 4. FEATURE IMPORTANCE FROM XGBOOST
xgb_model = models["XGBoost"]
xgb_model.fit(X_imputed, y)

# Extracting importance using feature_importances (weight metric)
importance_values = xgb_model.feature_importances_
importances_xgb = pd.Series(importance_values, index=feature_names).sort_values(ascending=False)

# Print Top 10 Features
print("\n============================")
print("TOP 10 FEATURES (XGBoost)")
print("============================")
top10_xgb = importances_xgb.head(10)
for i, (feature, importance) in enumerate(top10_xgb.items(), 1):
    print(f"{i}. {feature}: {importance:.4f}")

# Drawing a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(top10_xgb.index[::-1], top10_xgb.values[::-1])
plt.xlabel('Importance (weight)')
plt.title('Top 10 Feature Importance - XGBoost')
plt.tight_layout()
plt.savefig("feature_importance_xgb.tiff", dpi=300, format='tiff')
plt.show()
