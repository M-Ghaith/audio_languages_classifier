# Audio Language Classifier
A CNN-LSTM model that identifies spoken language from short audio clips across 6 European languages.

## Features
- Classifies audio into German, English, Spanish, French, Dutch, or Portuguese
- Extracts MFCC features (13 coefficients, 8kHz sample rate) from raw audio waveforms
- Hybrid CNN (spatial features) + LSTM (temporal sequences) architecture with batch normalization and dropout
- Includes PCA visualization of model outputs and TorchScript model export

## Tech Stack
- **Python** 3.x
- **PyTorch** + **torchaudio** — model training and audio transforms
- **NumPy** — data handling
- **scikit-learn** — train/test splitting and PCA
- **matplotlib** — loss and PCA plots
- **torchviz** — model architecture diagram

## Quick Start

```bash
# Install dependencies
pip install numpy torch torchaudio scikit-learn matplotlib torchviz

# Train the model (expects .npy data files in the project root)
python app.py

# Save the trained model as TorchScript
python save_model.py

# Visualize training/validation loss
python plot_train_val_loss.py

# Generate PCA plot of model outputs
python PCA.py
```

## Future Improvements
- [ ] Support variable-length audio sequences (adaptive pooling or LSTM-only architecture)
- [ ] Add an inference script for classifying a single audio file from the command line
- [ ] Include dataset download/preparation instructions and expected `.npy` file format
