import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


class RandomForestClassifierModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_prepare_data(self):
        # Load data
        self.data = pd.read_csv(self.data_path)
        
        # Drop unnecessary columns
        self.data.drop(columns=['Unnamed: 0', 'id', 'CustomerId', 'Surname'], inplace=True)

        # Handle missing values
        self.data.dropna(inplace=True)

        # Encode categorical data
        label_encoder = LabelEncoder()
        self.data['Gender'] = label_encoder.fit_transform(self.data['Gender'])
        
        # Preparing for One-Hot Encoding
        column_transformer = ColumnTransformer(
            [('one_hot_encoder', OneHotEncoder(), ['Geography'])],  
            remainder='passthrough'                                
        )

        # Apply One-Hot Encoding
        transformed_data = column_transformer.fit_transform(self.data)
        columns = ['Geography_France', 'Geography_Germany', 'Geography_Spain'] + \
                  [col for col in self.data.columns if col != 'Geography']
        
        # Convert numpy array back to DataFrame
        self.data = pd.DataFrame(transformed_data, columns=columns)

    def split_data(self):
        X = self.data.drop('churn', axis=1)
        y = self.data['churn']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        return accuracy, report
    
    def save_model(self, file_path):
        # Save the trained model to a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)

# Usage
if __name__ == "__main__":
    classifier = RandomForestClassifierModel('data_B.csv')
    classifier.load_and_prepare_data()
    classifier.split_data()
    classifier.train_model()
    accuracy, report = classifier.evaluate_model()
    print(f"Accuracy: {accuracy}\n")
    print(f"Classification Report:\n{report}")
    classifier.save_model('random_forest_model.pkl')  # Save the model
