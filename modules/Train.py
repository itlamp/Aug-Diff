from modules.Network import StlCNN

# imports for the tutorial
import numpy as np
import time
import os

# pytorch
import torch
import torch.nn as nn

def train_net(trainset, testset):
    # hyper-parameters
    batch_size = 32
    learning_rate = 1e-4
    epochs = 50

    # dataloaders - creating batches and shuffling the data
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # device - cpu or gpu?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loss criterion
    criterion = nn.CrossEntropyLoss()

    # build our model and send it to the device
    model = StlCNN().to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # function to calcualte accuracy of the model
    def calculate_accuracy_and_loss(model, dataloader, device, criterion):
        model.eval() # put in evaluation mode
        total_correct = 0
        total_images = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total_images += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                running_loss += loss.data.item()
            running_loss /= len(trainloader)
        model_accuracy = total_correct / total_images * 100
        return model_accuracy, running_loss


    # training loop
    for epoch in range(1, epochs + 1):
        model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time()
        for _ , data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # send them to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # calculate the loss
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters

            # print statistics
            running_loss += loss.data.item()

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        # Calculate training/test set accuracy of the existing model
        train_accuracy, _ = calculate_accuracy_and_loss(model, trainloader, device, criterion)
        test_accuracy, test_loss = calculate_accuracy_and_loss(model, testloader, device, criterion)

        log = "Epoch: {} | Training Loss: {:.4f} | Test Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch, running_loss, test_loss, train_accuracy, test_accuracy)
        # with f-strings
        # log = f"Epoch: {epoch} | Loss: {running_loss:.4f} | Training accuracy: {train_accuracy:.3f}% | Test accuracy: {test_accuracy:.3f}% |"
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        # with f-strings
        # log += f"Epoch Time: {epoch_time:.2f} secs"
        print(log)
        
    print('==> Finished Training ...')