# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale the data because Multinomial and Complement require non-negative values
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Multinomial Naive Bayes

multinomial = MultinomialNB()
multinomial.fit(X_train_scaled, y_train)

y_pred_multi = multinomial.predict(X_test_scaled)

print("MultinomialNB Accuracy:", accuracy_score(y_test, y_pred_multi))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_multi))


# Complement Naive Bayes

complement = ComplementNB()
complement.fit(X_train_scaled, y_train)

y_pred_comp = complement.predict(X_test_scaled)

print("ComplementNB Accuracy:", accuracy_score(y_test, y_pred_comp))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_comp))


# Bernoulli Naive Bayes

# Convert scaled data into binary form (0 or 1)
binarizer = Binarizer(threshold=0.5)
X_train_binary = binarizer.fit_transform(X_train_scaled)
X_test_binary = binarizer.transform(X_test_scaled)

bernoulli = BernoulliNB()
bernoulli.fit(X_train_binary, y_train)

y_pred_bern = bernoulli.predict(X_test_binary)

print("BernoulliNB Accuracy:", accuracy_score(y_test, y_pred_bern))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_bern))
