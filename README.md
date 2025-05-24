Wine Quality Prediction with XGBoost model

This project leverages the Wine Quality Dataset from the UCI Machine Learning Repository to build an ML pipeline for predicting wine quality scores.  
https://archive.ics.uci.edu/ml/datasets/wine+quality  
The workflow covers model building, containerization, and deployment on Heroku.

Dataset
The dataset contains several physicochemical properties of wine samples and a quality score (between 0 and 10). The 11 features used as input for the model are:  
    -fixed acidity  
    -volatile acidity  
    -citric acid  
    -residual sugar  
    -chlorides  
    -free sulfur dioxide  
    -total sulfur dioxide  
    -density  
    -pH  
    -sulfates  
    -alcohol  

Project Structure

    notebooks/: Data analysis and model training .ipynb
    templates/: Web app HTML template
    runtime.txt: py version
    app.py: Flask API for model inference
    requirements.txt: Dependencies
    model.pkl: Trained XGB model
    Dockerfile: For containerization
    Procfile: For Heroku deployment
