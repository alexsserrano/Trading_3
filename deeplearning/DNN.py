from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Modelo DNN
model_dnn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
