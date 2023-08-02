QSPRmodeler - an open source application for molecular predictive analytics
==============================

# The introduction
Quantitative Structure-Property Relationship (QSPR) is a computational approach that uncovers associations between a structure representation and a target molecular characteristic. The structure can be represented using molecular descriptors and/or molecular fingerprints, and their correlation with the desired target feature is investigated.

Several machine learning methodologies can be employed within this framework, including XGBoost, Artificial Neural Networks (ANN), Random Forest, Bagging, Support Vector Machines (SVM), Ridge, and others. These methodologies can be utilized to build regression or classification models.

The provided library serves the purpose of processing data and optimizing the hyperparameters of the machine learning models to identify the most suitable one. Depending on the problem being addressed, the target feature undergoes specific processing. For classification issues, binarization is performed, with the user defining the threshold for the target feature. For regression problems, logarithmic scaling is applied.

If molecular fingerprints are used as structural features, Principal Component Analysis (PCA) can be employed, which is particularly essential when using MLP.

To employ the library, users select a specific methodology for examination and execute the training procedure. The outcome is the saved best model, determined by the optimal hyperparameters, exhibiting the highest agreement between real and predicted values during training and testing.

The library proves valuable for exploring various machine learning methodologies in QSPR or QSAR studies. Leveraging open-source chemoinformatic libraries like RDKit or Mordred, the entire library is implemented in Python.

# How to use the software
1. Place your dataset with the SMILES column and the target feature inside of the ```data``` folder.

2. Define the methodology and the problem type; you might inspire with one of the files present in the ```training_configurations``` folder. There are also quite a fe other settings you might to adjust.
```json
{
    "prediction_methodology": "M_MLP",
    "prediction_type": "regression",
    "GPU_support": false,
    "training_aux_data": 
    {
        "run_name": "test", 
        "n_outer": 1, 
        "n_cv": 5, 
        "development": false, 
        "goal_function": "mse", 
        "goal_function_multiplier": 1.0, 
        "threshold": null,
        "track_model": false, 
        "experiment": null, 
        "comment": null, 
        "max_evals": 2,
        "pipeline_file": "ar/pipeline_configuration_xgboost_regression_Morgan_1024_pca_50_QED.json"
    },
    "data_preparation": {
        "data_file": "ar_homo_sapiens_chembl_3.3_more_ChEMBL_columnsIC50EC50_combined_expert.csv",
        "molecule_column": "coms_canonical_smiles",
        "experimental_data_prefix": "act_median_expert",
        "target_column": "experimental_value",
        "max_level_activity": 1,
        "std_threshold": 100,
        "strategy": "median"
    },
    "model_storage": {
        "resulting_model": "ar_mlp_regressor_pca50_expert.model"
    }
}
```

3. Define the data processing workflow, inspire with one of the files from the ```pipelines``` directory, e.g.:
```json
{
    "feature_transform":
    {
        "Calculate_molecular_features":
        {
            "proceed": "yes",
            "calculate_fps": "no",
            "calculate_descriptors": "yes",
            "fingerprint_size": 1024,
            "fp_type": "morgan",
            "descriptors_types": ["QED", "Lipinski"],
            "meta_descriptors_types": ["QED"],
            "label": "Molecular features"
        },
        "Scaling":
        {
            "proceed": "yes",
            "scaling_features": ["SLogP", "SMR", "naRing", "nHBAcc", "nHBDon", "nRot", "MW", "TopoPSA", "QED"],
            "label": "Scaling"
        },
        "PCA":
        {
            "proceed": "no",
            "apply_to_md": "no",
            "apply_to_fp": "yes",
            "n_components": 20,
            "label": "PCA"
        }
    },
    "target_transform":
    {
        "Target_transform":
        {
            "proceed": "no"
        },
        "Target_binarization":
        {
            "proceed": "yes",
            "threshold": 1000,
            "label": "Target Binarization"
        },
        "O_removal":
        {
            "proceed": "no",
            "factor": 1.5,
            "label": "Outlier removal"
        }
    }
}
```

4. Now run the script ```QSPRmodeler_train.py``` (inside of the ```scripts``` folder) with an obligatory argument ```--training_conf``` pointing to the training configuration file.

```bash
python activity_prediction_model_creation.py --training_conf <training configuration file.json
```
The training will start, and all the parameters that were previously set will be applied during the code execution.

5. The best model can be loaded, and predictions can be made.

6. To show how the trained model can be used in the inference mode we delver a set of Jupyter notebooks (directory: ```notebooks```).