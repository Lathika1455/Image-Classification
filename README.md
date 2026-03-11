# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Image classification is a fundamental problem in computer vision, where the goal is to assign an input image to one of the predefined categories. Traditional machine learning models rely heavily on handcrafted features, whereas Convolutional Neural Networks (CNNs) automatically learn spatial features directly from pixel data.

In this experiment, the task is to build a Convolutional Deep Neural Network (CNN) to classify images from the FashionMNIST dataset into their respective categories. The trained model will then be tested on new/unseen images to verify its effectiveness.

## Neural Network Model

<img width="1027" height="441" alt="image" src="https://github.com/user-attachments/assets/ba484496-51cc-48cb-bb12-6359709efe49" />


## DESIGN STEPS

## STEP 1:
Load Fashion-MNIST dataset from torchvision, apply transformations, and create DataLoaders for batch processing

## STEP 2:
Build CNN architecture with 3 convolutional layers (32,64,128 filters) and 3 fully connected layers (128,64,10 nodes)

## STEP 3:
Train model using CrossEntropyLoss and Adam optimizer while tracking training and validation loss metrics

## STEP 4:
Evaluate model performance using confusion matrix, classification report, and test on new handwritten images

## STEP 5:
Visualize results with loss plots and display predictions with actual vs predicted labels

## PROGRAM

### Name: LATHIKA SREE R
### Register Number: 212224040169
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # Convolution Layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Convolution Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))   # 28x28 → 14x14
        x = self.pool(self.relu(self.conv2(x)))   # 14x14 → 7x7

        x = x.view(-1, 64 * 7 * 7)                # Flatten

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```python
def train_model(model, train_loader, num_epochs=3):

    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            # Move to GPU if available
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: LATHIKA SREE R')
        print('Register Number: 212224040169')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT
### Training Loss per Epoch

<img width="301" height="208" alt="image" src="https://github.com/user-attachments/assets/b234a8ab-e9ea-404f-8e15-10255d00ae9d" />


### Confusion Matrix

<img width="898" height="873" alt="image" src="https://github.com/user-attachments/assets/2a4f4a4a-7eb5-490f-af3b-16c809e45817" />

### Classification Report

<img width="550" height="420" alt="image" src="https://github.com/user-attachments/assets/0030f107-b1fd-4472-9032-f03ab64ea83e" />

### New Sample Data Prediction

<img width="499" height="611" alt="image" src="https://github.com/user-attachments/assets/3d7b8f7e-bffb-4024-9b71-cddfbeacb599" />


## RESULT

Successfully developed and trained a CNN model on Fashion-MNIST dataset achieving good classification accuracy across 10 fashion categories with proper loss visualization and validation on test images.
