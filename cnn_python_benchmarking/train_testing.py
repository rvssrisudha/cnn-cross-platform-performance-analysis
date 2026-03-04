# Training
def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        # Ouput of feedforwarding
        output = model(data)

        # Loss calibration
        loss = F.nll_loss(output, target)

        # Gradient
        loss.backward()

        # Back propagation
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Test
def test():
    model.eval()

    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)

        # Output of feedforwarding
        output = model(data)

        test_loss += F.nll_loss(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
# Traning process
for epoch in range(1, 10):
    train(epoch)
    test()
