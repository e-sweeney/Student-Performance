Student-Performance
=================
***** Changed repo to use ArgoCD

Identify students grades at college based on their hours studied, previous scores, hours sleep etc.

An index.html input form is available which will input the features required.
A result.html page is available which will display the result.
Both these pages should be placed in a templates directory as you are using the render_template function.

app.py will serve the model.

trainBackup.py will train the model and produce a model.pkl file - old file

train.py trains the regression model and records metric and model with mlflow.  Please ensure mlflow is running.

Student_Performance.csv is the dataset required.

