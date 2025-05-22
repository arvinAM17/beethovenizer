import torch


def train_epoch(model, dataloader, optimizer, criterion, device, epoch_num, writer=None):
    """
    Train the model for one epoch.
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to train on.
        writer (SummaryWriter, optional): TensorBoard writer. Defaults to None.
    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_ind, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        output, _ = model(inputs)
        loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predicted = output.argmax(dim=-1)
        mask = targets != model.pad_idx  # ignore padding

        total_correct += ((predicted == targets) & mask).sum().item()
        total_samples += mask.sum().item()

        if (batch_ind + 1) % 100 == 0:
            accuracy = total_correct / total_samples

            print(
                f"Batch {batch_ind + 1}/{len(dataloader)}; Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

            if writer:
                writer_ind = batch_ind + 1 + epoch_num * len(dataloader)
                writer.add_scalar('Loss/training', loss.item(), writer_ind)
                writer.add_scalar('Accuracy/training', accuracy, writer_ind)

    accuracy = total_correct / total_samples
    loss = total_loss / len(dataloader)

    return loss, accuracy


def train_model(model, dataloader, optimizer, criterion, device, epochs=10, writer=None):
    """
    Train the model for multiple epochs.
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to train on.
        epochs (int, optional): Number of epochs to train. Defaults to 10.
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        avg_loss, avg_accuracy = train_epoch(model, dataloader, optimizer,
                                             criterion, device, epoch, writer)
        print(
            f"Epoch {epoch + 1}/{epochs}; Average Loss: {avg_loss:.4f} - Average Accuracy: {avg_accuracy:.4f}")

        if writer:
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Accuracy/train', avg_accuracy, epoch)
