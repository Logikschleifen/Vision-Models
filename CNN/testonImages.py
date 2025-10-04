from PIL import Image
import torch
from torchvision import transforms

# 2a. Prepare transform matching training time
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2b. Load model and weights
model = FashionClassifier()  # or your CNN class
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 2c. Load saved images and classify
image_folder = 'fashion_images'
for fname in os.listdir(image_folder):
    if not fname.lower().endswith('.png'):
        continue
    path = os.path.join(image_folder, fname)
    img = Image.open(path).convert('L')  # FashionMNIST is grayscale, “L” mode
    # If your model expects shape (1,28,28), ensure correct size
    img_t = transform(img).unsqueeze(0)  # shape [1, 1, 28, 28]
    img_t = img_t.to(device)
    
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, dim=1)
        print(fname, "=>", pred.item())
