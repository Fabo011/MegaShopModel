### Create Environment
```bash
python3 -m venv path/to/venv
source path/to/venv/bin/activate
```

---

### Install Requirements
```bash
pip3 install -r requirements.txt
```

### Start Training
```bash
python3 main.py
```

---

### Data
https://gist.github.com/ryanorsinger/cb1222e506c1266b9cc808143ddbab82

---

### Description
**KMeans Model (kmeans_model.pkl):** 
This model contains the clustering algorithm that groups customers into clusters based on their features (age, income, spending score). It is used to predict the cluster assignments for new data.

**Scaler Model (scaler.pkl):** 
This model standardizes the input features to ensure that they have a mean of 0 and a standard deviation of 1. This step is crucial for clustering algorithms like K-Means, which are sensitive to the scale of the data. The scaler ensures that new data is transformed in the same way as the training data.