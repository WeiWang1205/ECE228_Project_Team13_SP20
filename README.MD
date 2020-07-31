# ECE228Project_Team13_SP20

### Dataset "UrbanSound8K" link: wget https://goo.gl/8hY5ER


## File Organization 
### Datasets/ ----------- Contains dataset API.<br>
There is a dataset.py and resnet.py file that defines the dataset API in the "Datasets" fold. The dataset class can automatically load, preprocess and generate data batch for training and testing. And the resnet define the resnet layers structure and implementation<br>

### Deep_learning_methods/ ------- Contains results of deep machine learning methods.<br>
There are three files in the "Deep_learning_methods" fold corresponding to RNN, CNN and DNN methods. Using the pytorch network models, we provide the accuracy and loss finally. Each file has only one fold results as demo instead of taking 10-fold cross validation.

### Traditional_methods/ ------- Contains results of traditional machine learning methods.<br>
There are four files in the "traditional_methods" fold -- two for mel spectrogram feature of lower dimension and two for mfcc feature of lower dimension. Each file applies four traditonal methods which are SVM, KNN, RF and XGBoost and has only one fold results as demo instead of taking 10-fold cross validation.

