# Save model
torch.save(model, "./cnn_mnist.pt")

# Load model
model = torch.load("./cnn_mnist.pt", weights_only=False)
model.eval()
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

from PIL import Image
# Load bitmap image
img = Image.open("train_0.bmp", "r")
np_img = np.array(img)
pyplot.imshow(np_img, cmap=pyplot.get_cmap('gray'))

np_img_re = np.reshape(np_img, (1,1,28,28))
# 0 - 255 => 0 - 1 , np.array => tensor 
data = Variable(torch.tensor((np_img_re / 255), dtype = torch.float32))

# Output of feedforwarding
output = model(data)
pred = output.data.max(1, keepdim=True)[1]
print('Predicted output: ' + ', '.join(map(str, pred.flatten().tolist())))


