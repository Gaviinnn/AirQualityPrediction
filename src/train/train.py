import torch
from torch.optim.lr_scheduler import StepLR

def train_model(model, train_loader, criterion, optimizer, epochs, device, patience=5, step_size=10, gamma=0.5):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        average_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {average_loss:.4f}")

        # Early stopping
        if average_loss < best_loss:
            best_loss = average_loss
            patience_counter = 0
            torch.save(model.state_dict(), "./outputs/model_checkpoint/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
