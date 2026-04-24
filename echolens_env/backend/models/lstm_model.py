import tensorflow as tf
from tensorflow.keras import layers, models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMSignLanguageModel:
    def __init__(self, input_shape: tuple, num_classes: int, dropout_rate: float = 0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        logger.info(f"LSTMSignLanguageModel initialized")
    
    def build_model(self):
        logger.info("Building LSTM model...")
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.LSTM(256, return_sequences=True, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.LSTM(128, return_sequences=True, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.LSTM(64, return_sequences=False, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        self.model = model
        logger.info("✅ LSTM model built")
        return model
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy'):
        if self.model is None:
            self.build_model()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        logger.info("✅ Model compiled")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model."""
        if self.model is None:
            self.build_model()
            self.compile_model()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7
                )
            ]
        )
        
        return history
    
    def get_summary(self):
        if self.model:
            self.model.summary()
    
    def save_model(self, filepath: str):
        if self.model:
            # Ensure filename ends with .weights.h5
            if not filepath.endswith('.weights.h5'):
                filepath = filepath.replace('.h5', '.weights.h5')
            self.model.save_weights(filepath)
            logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        if self.model is None:
            self.build_model()
        # Handle both .h5 and .weights.h5 formats
        if not filepath.endswith('.weights.h5'):
            filepath = filepath.replace('.h5', '.weights.h5')
        self.model.load_weights(filepath)
        logger.info(f"✅ Model loaded from {filepath}")